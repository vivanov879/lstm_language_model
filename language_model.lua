require 'mobdebug'.start()
require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'Embedding'    
require 'lstm'
local model_utils=require 'model_utils'



function read_words(fn)
  fd = io.lines(fn)
  sentences = {}
  line = fd()

  while line do
    sentence = {}
    for _, word in pairs(string.split(line, " ")) do
        sentence[#sentence + 1] = word
    end
    sentences[#sentences + 1] = sentence
    line = fd()
  end
  return sentences
end


function convert2tensors(sentences)
  local t = torch.Tensor(#sentences, #sentences[1])
  for k, sentence in pairs(sentences) do
    assert(#sentence == #sentences[1])
    for i = 1, #sentence do 
      t[k][i] = tonumber(sentence[i])
    end
  end
  return t  
end

function calc_max_sentence_len(sentences)
  local m = 1
  for _, sentence in pairs(sentences) do
    m = math.max(m, #sentence)
  end
  return m
end




vocabulary_raw = read_words('vocabulary_raw')
inv_vocabulary_raw = read_words('inv_vocabulary_raw')



vocabulary = {}
inv_vocabulary = {}

for i, sentence in pairs(vocabulary_raw) do 
  vocabulary[tonumber(sentence[1])] = sentence[2]
  inv_vocabulary[sentence[2]] = tonumber(sentence[1])
end

vocabulary[#vocabulary + 1] = 'EOSMASK'
inv_vocabulary['EOSMASK'] = #vocabulary
vocab_size = #vocabulary

x_train_raw = read_words('x_train_sorted1')
y_train_raw = read_words('y_train_sorted1')

x_train_lens = torch.Tensor(#x_train_raw)
for i, sentence in pairs(x_train_raw) do 
  x_train_lens[i] = #sentence
end

sorted, indexes = torch.sort(x_train_lens, 1)

x_train = {}
y_train = {}
for i = 1, indexes:size(1) do
  x_train[#x_train + 1] = x_train_raw[indexes[i]]
  y_train[#y_train + 1] = y_train_raw[indexes[i]]
end

--[[
fd = io.open('x_train_sorted', 'w')
for _, sentence in pairs(x_train) do
  fd:write(table.concat(sentence, ' ') .. '\n')
end

fd = io.open('y_train_sorted', 'w')
for _, sentence in pairs(y_train) do
  fd:write(table.concat(sentence, ' ') .. '\n')
end
]]--

x_dev_raw = read_words('x_train_sorted1')
y_dev_raw = read_words('y_train_sorted1')

max_sentence_len = math.max(calc_max_sentence_len(x_dev_raw), calc_max_sentence_len(x_train_raw))

batch_size = 4
n_data = #x_train_raw
data_index = 1


function gen_batch()
  end_index = data_index + batch_size
  if end_index > n_data then
    end_index = n_data
    data_index = 1
  end
  start_index = end_index - batch_size
  
  function f(sentences)
    local t = torch.zeros(batch_size, max_sentence_len)
    local mask = torch.zeros(max_sentence_len, batch_size, batch_size)
    local max_sentence_len_batch = 1
    for k = 1, batch_size do
      local sentence = sentences[start_index + k - 1]
      max_sentence_len_batch = math.max(max_sentence_len_batch, #sentence)
      for i = 1, max_sentence_len do 
        if i <= #sentence then
          t[k][i] = sentence[i]
          mask[i][k][k] = 1
        else
          t[k][i] = vocab_size
          mask[i][k][k] = 0
        end
      end
    end
    return t[{{}, {1, max_sentence_len_batch}}], mask[{{1, max_sentence_len_batch},{},{}}]
  end
  
  local batch_x, mask_x = f(x_train)
  local batch_y, mask_y = f(y_train)
  
  data_index = data_index + 1
  if data_index > n_data then 
    data_index = 1
  end
  
  return batch_x, mask_x, batch_y, mask_y
  
end


opt = {}
rnn_size = 100
seq_length = max_sentence_len
opt.rnn_size = rnn_size

x_raw = nn.Identity()()
x = Embedding(vocab_size, rnn_size)(x_raw)

prev_c = nn.Identity()()
prev_h = nn.Identity()()
next_h, next_c = make_lstm_step(opt, x, prev_h, prev_c)

z = nn.Linear(rnn_size, vocab_size)(next_h)
prediction = nn.LogSoftMax()(z)

lstm = nn.gModule({x_raw, prev_c, prev_h}, {next_c, next_h, prediction})


criterion = nn.ClassNLLCriterion()


-- put the above things into one flattened parameters tensor
local params, grad_params = model_utils.combine_all_parameters(lstm)
params:uniform(-0.08, 0.08)

-- make a bunch of clones, AFTER flattening, as that reallocates memory
lstm_clones = model_utils.clone_many_times(lstm, seq_length)
criterion_clones = model_utils.clone_many_times(criterion, seq_length)


-- LSTM initial state (zero initially, but final state gets sent to initial state when we do BPTT)
local initstate_c = torch.zeros(batch_size, rnn_size)
local initstate_h = initstate_c:clone()

-- LSTM final state's backward message (dloss/dfinalstate) is 0, since it doesn't influence predictions
local dfinalstate_c = initstate_c:clone()
local dfinalstate_h = initstate_c:clone()

-- do fwd/bwd and return loss, grad_params
function feval(params_)
    if params_ ~= params then
        params:copy(params_)
    end
    grad_params:zero()
    
    x, mask_x, y, mask_y  = gen_batch()
    
    ------------------- forward pass -------------------
    local lstm_c = {[0]=initstate_c} -- internal cell states of LSTM
    local lstm_h = {[0]=initstate_h} -- output values of LSTM
    prediction = {}           -- softmax outputs
    local loss = 0

    for t=1, x:size(2) do
      lstm_c[t], lstm_h[t], prediction[t]  = unpack(lstm_clones[t]:forward({x[{{}, t}], lstm_c[t-1], lstm_h[t-1]}))
      prediction[t] = torch.mm(mask_x[t], prediction[t])
      loss = loss + criterion_clones[t]:forward(prediction[t], y[{{}, t}])
    end

    ------------------ backward pass -------------------
    -- complete reverse order of the above
    local dlstm_c = {[x:size(2)]=dfinalstate_c}
    local dlstm_h = {[x:size(2)]=dfinalstate_c}
    local dprediction = {}
    local dx = {}
    
    for t=x:size(2),1,-1 do
      dprediction[t] = criterion_clones[t]:backward(prediction[t], y[{{}, t}])
      dprediction[t] = torch.mm(mask_x[t], dprediction[t])
      dx[t], dlstm_c[t-1], dlstm_h[t-1] = unpack(lstm_clones[t]:backward({x[{{}, t}], lstm_c[t-1], lstm_h[t-1]}, {dlstm_c[t], dlstm_h[t], dprediction[t]}))
    end

    ------------------------ misc ----------------------
    -- transfer final state to initial state (BPTT)
    initstate_c:copy(lstm_c[#lstm_c])
    initstate_h:copy(lstm_h[#lstm_h])

    -- clip gradient element-wise
    grad_params:clamp(-5, 5)

    return loss, grad_params
end

-- optimization stuff
local losses = {}
local optim_state = {learningRate = 1e-1}
for i = 1, 1000 do
    local _, loss = optim.adagrad(feval, params, optim_state)
    losses[#losses + 1] = loss[1]
    if i % 1 == 0 then
      
      sample_sentence = {}
      target_sentence = {}
      
      for t = 1, x:size(2) do 
        _, sampled_index = prediction[t]:max(2)
        --print(sampled_index)
        sample_sentence[#sample_sentence + 1] = vocabulary[sampled_index[1][1]]
        target_sentence[#target_sentence + 1] = vocabulary[y[1][t]]
     end
    print(sample_sentence)
    print(target_sentence)
    print(string.format("iteration %4d, loss = %6.8f, loss/seq_len = %6.8f, gradnorm = %6.4e", i, loss[1], loss[1] / seq_length, grad_params:norm()))
    end
    
    
    if i % 100 == 0 then
        torch.save('lstm_model', lstm)
    end

end



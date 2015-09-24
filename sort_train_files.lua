
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



x_train_raw = read_words('x_train')
y_train_raw = read_words('y_train')


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

fd = io.open('x_train_sorted', 'w')
for _, sentence in pairs(x_train) do
  fd:write(table.concat(sentence, ' ') .. '\n')
end

fd = io.open('y_train_sorted', 'w')
for _, sentence in pairs(y_train) do
  fd:write(table.concat(sentence, ' ') .. '\n')
end

require 'mobdebug'.start()
require 'nn'
require 'nngraph'
require 'optim'
require 'Embedding'
local model_utils=require 'model_utils'
require 'table_utils'
nngraph.setDebug(true)



x = torch.rand(3,5)
print(x)

sorted, indexes = torch.sort(x:sum(2), 1)
print(x:sum(2))

y = x:index(1, indexes[{{}, 1}])
print(y)

dummy_pass = 1



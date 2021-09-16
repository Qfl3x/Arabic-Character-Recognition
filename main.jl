using CSV
using DataFrames
using Plots
using Flux: onehot,onehotbatch,onecold
using Flux: crossentropy, Statistics.mean
using Flux
using CUDA
using MLDataPattern:splitobs, shuffleobs

train_data_input = CSV.read("csvTrainImages 13440x1024.csv", DataFrame)
test_data_input = CSV.read("csvTestImages 3360x1024.csv", DataFrame)

train_label_input = CSV.read("csvTrainLabel 13440x1.csv", DataFrame)
test_label_input = CSV.read("csvTestLabel 3360x1.csv", DataFrame)

function get_data()

    train_data = reshape(Array(train_data_input),:,32,32)/255;
    test_data = reshape(Array(test_data_input),:,32,32)/255;
    train_data = reverse(train_data,dims=(2));
    test_data = reverse(test_data,dims=(2));

    train_label = reshape(onehotbatch(Array(train_label_input),1:28),28,13439);
    test_label = reshape(onehotbatch(Array(test_label_input),1:28),28,3359);

    train_data = permutedims(train_data,[2,3,1])
    train_data = reshape(train_data,32,32,1,13439)

    test_data = permutedims(test_data,[2,3,1])
    test_data = reshape(test_data,32,32,1,3359)

    #Validation

    (data,label) = shuffleobs((train_data,train_label))
    (train_data,train_label), (val_data,val_label) = splitobs((data,label), at = 0.85)

    train_loader = Flux.DataLoader((data=Float64.(train_data),label=Float64.(train_label)),batchsize=128,shuffle=true)
    val_loader = Flux.DataLoader((data=Float64.(val_data),label=Float64.(val_label)),batchsize=128)

    return train_loader, test_data, test_label, val_loader
end

function network()
  return Chain(

    Conv((3, 3), 1=>16, pad=(1,1), relu),
    x -> maxpool(x, (2,2)), #16x16
    BatchNorm(16),

    Conv((3, 3), 16=>32, pad=(1,1), relu),
    x -> maxpool(x, (2,2)), #8x8
    BatchNorm(32),


    Conv((3, 3), 32=>64, pad=(1,1), relu),
    x -> maxpool(x, (2,2)), # 4x4
    BatchNorm(64),

    Flux.flatten,
    Dense(1024,256),
    Dropout(0.2),
    Dense(256,28),


    softmax,)
end


function train_model(epochs)
  model = network()
  model = gpu(model)

  loss(x,y) = crossentropy(model(x),y)
  ps = params(model)
  opt = ADAM(0.0005)

  val_error = zeros(epochs)
  train_error = zeros(epochs)

  val_error_current = 0.
  train_error_current = 0.

  for epoch in 1:epochs
    for (x,y) in train_loader
      x =  gpu(x)
      y =  gpu(y)
      gs = Flux.gradient(() -> loss(x,y),ps)
      train_error_current += loss(x,y)
      Flux.update!(opt,ps,gs)
    end
    for (x,y) in val_loader
      x =  gpu(x)
      y =  gpu(y)
      val_error_current += loss(x,y)
    end

    train_error_current /= length(train_loader)
    val_error_current /= length(val_loader)

    val_error[epoch] = val_error_current
    train_error[epoch] = train_error_current

    println("Epoch: ", epoch)
    println("Validation error: ", val_error_current)
    println("Training error: ", train_error_current)
    println("===========================")
    val_error_current = 0.
    train_error_current = 0.
  end

  return model,train_error,val_error
end

train_loader,test_data,test_label,val_loader = get_data()
model, train_error, val_error = train_model(30)

#Plot:
plt = plot(5:30,train_error[5:end],label="training",xlabel="Epochs",ylabel="Loss",legend=:bottomleft)
plot!(5:30,val_error[5:end],label="validation")

#Accuracy:
m = cpu(model)
accuracy(x,y) = mean(onecold(m(x)) .== onecold(y))
accuracy(Float64.(test_data),Float64.(test_label))

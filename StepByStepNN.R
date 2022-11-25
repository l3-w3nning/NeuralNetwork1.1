
# data <- data.frame(x=runif(1000),y=runif(1000)) %>% 
#         mutate(label=ifelse(test=sqrt((x**2+y**2))<=0.5,yes = "red" ,no="blue")) 
# 
# data %>% ggplot(aes(x=x,y=y,colour=label))+geom_point()+theme_bw()

classifier <- function(point_x,point_y,weight_matrix,bias){
  point_xy=c(point_x,point_y)
  res <- weight_matrix%*%point_xy+bias
  return (ifelse(res[1]>=res[2],1,0)) #binary classifier, later used for contour
}

get_weight_matrix_and_bias <- function(w_11,w_12,w_21,w_22,b_1,b_2){
  return(c(mat=matrix(c(w_11,w_12,w_21,w_22),nrow=2,ncol=2),bias=c(b_1,b_2)))
}



decision_boundary <- function(weight_matrix,bias){
  return(c(m=(weight_matrix[2,2]-weight_matrix[1,2])/(weight_matrix[1,1]-weight_matrix[1,2]),b=bias[2]-bias[1]))
}

library(methods)

layer <- setRefClass("Layer", fields = list(num_nodes_in = "numeric", 
                                              num_nodes_out = "numeric",weights = "matrix", bias = "numeric"), methods = list(
                                                initialize = function(num_nodes_in,num_nodes_out,weights,bias){
                                                  .self$num_nodes_in <- num_nodes_in
                                                  .self$num_nodes_out <- num_nodes_out
                                                  mat=matrix(weights,nrow=num_nodes_in,ncol=num_nodes_out)
                                                  .self$weights = mat
                                                  .self$bias = bias
                                                },
                                                CalcOutput = function(input)
                                                {
                                                  
                                                  output =.self$bias
                                                  output = output + input %*% .self$weights
                                                  
                                                  print(output)
                                                  return (output)
                                                }
                                              ))


NeuralNetwork <- setRefClass("NN",fields = list(layer_sizes = "numeric",num_nodes_out="numeric",weights = "list",bias = "list"),
                                            #initialize nn by creating list of layers (with given layer sizes)
                                            #then write a method which runs inputs through layers (CalcOutput)
                                            #weights ist hier eine list of lists
                                            #biases ist hier auch eine list of lists
                                            methods = list(
                                                  intialize = function(layer_sizes,weights,bias){
                                                   .self$layer_sizes <- layer_sizes
                  
                                                  },
                                                  
                                                  GetLayers = function(){ 
                                                    layers=list()
                                                    for (i in 1:(length(.self$layer_sizes)-1)){
                                                      layers <- append(layers,layer$new(num_nodes_in=.self$layer_sizes[i],num_nodes_out=.self$layer_sizes[i+1],weights=.self$weights[[i]],bias=.self$bias[[i]]))
                                                    }
                                                    return(layers)
                                                  },
                                                  
                                                  CalcOutputNN = function(inputx,inputy)
                                                  {
                                                    input = c(inputx,inputy)
                                                    layers = .self$GetLayers()
              
                                                    for (layer in layers){
                                                      input = layer$CalcOutput(input)
                                                      
                                                    }
                                                    output=input
                                                    return (output)
                                                  },
                                                  Classify = function(output){
                                                    return(ifelse(output[1]>=output[2],1,0))
                                                  }
                                                ))


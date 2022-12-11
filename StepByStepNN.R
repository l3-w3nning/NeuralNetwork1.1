

library(methods)

layer <- setRefClass("Layer", fields = list(num_nodes_in = "numeric", 
                                              num_nodes_out = "numeric",weights = "matrix", bias = "numeric",weight_gradient = "matrix",bias_gradient="numeric"), methods = list(
                                                initialize = function(num_nodes_in,num_nodes_out,weights,bias,weight_gradients=NA,bias_gradients=NA){
                                                  .self$num_nodes_in <- num_nodes_in
                                                  .self$num_nodes_out <- num_nodes_out
                                                  mat=matrix(weights,nrow=num_nodes_in,ncol=num_nodes_out)
                                                  .self$weights = mat
                                                  .self$bias = bias
                                                  if (is.na(weight_gradients) && is.na(bias_gradients)){#if weight gradient and bias gradient not given in initialization, set to zero
                                                    .self$weight_gradient = matrix(0L, nrow = num_nodes_in, ncol = num_nodes_out) 
                                                    .self$bias_gradient = rep(0,length(bias))
                                                  }
                                                  else {
                                                    .self$weight_gradient = weight_gradients
                                                    .self$bias_gradient = bias_gradients
                                                  }
                                                },
                                                ActivationFunction = function(output){ #sigmoid activation function, later i will also implement backpropagation analytically (shape of activation fct. will then become important)
                                                  return (sapply(output,FUN=function(x) 1/(1+exp(-x))))
                                                },
                                                CalcOutput = function(input,bias,weights)
                                                {
              
                                                  output = bias
                                                  output = output + input %*% weights
        
                                                  return (.self$ActivationFunction(output))
                                                }
                                              ))


NeuralNetwork <- setRefClass("NN",fields = list(layer_sizes = "numeric",num_nodes_nn="list",weights = "list",bias = "list",layers="list",weight_gradient="list",bias_gradient="list"),
                                            
                                            methods = list(
                                                  intialize = function(layer_sizes,num_nodes_nn,layers=NA,weight_gradient=NA,bias_gradient=NA){
                                                   .self$layer_sizes <- layer_sizes
                                                   .self$num_nodes_nn <- num_nodes_nn
                                                   .self$weights <- weights
                                                   .self$bias <- bias
                                                   if (is.na(weight_gradient) && is.na(bias_gradient)){
                                                     .self$weight_gradient <- list(NA,NA)
                                                     .self$bias_gradient <- list(NA,NA)
                                                   }
                                                   
                                                   if (is.na(layers)){
                                                     .self$layers <- list(NA,NA)
                                                   }
                                                  },
                                                  
                                                  GetLayers = function(){
                                                    layer_list=list()
                                                    for (i in 1:(length(.self$layer_sizes)-1)){
                                                      layers_list <- append(layers,layer$new(num_nodes_in=.self$layer_sizes[i],num_nodes_out=.self$layer_sizes[i+1],weights=.self$weights[[i]],bias=.self$bias[[i]]))
                                                    }
                                                    return(layer_list)
                                                  },

                                                  CalcOutputNN = function(inputx,inputy)
                                                  {
                                                    input = c(inputx,inputy)
                                                    
                                                    # if (is.na(.self$layers)){
                                                    #   layers_list = .self$GetLayers()  #set layers to default if no layers are given as input
                                                    # 
                                                    # }
                                                    # else{
                                            
                                                      for (layer_depth in 1:length(.self$layers)){
                                                        
                                                        input = input %*% .self$layers[[layer_depth]]$weights
                                                        input = input + .self$layers[[layer_depth]]$bias
                                                        input = .self$ActivationFunction(input)
                                                        
                                                      }            
                                                    #}
                                          
                                                    output=input
                                                    return (output)
                                                  },
                                                  Classify = function(output){
                                                    return(ifelse(output[1]>=output[2],1,0))
                                                  },
                                                  
                                                  Learn = function(data_train,learnrate,stepwidth){
                                                    
                                                    layer_depth=1
                                                    for (layer in .self$layers){
                                                      
                                                      output_train = mapply(.self$CalcOutputNN,data_train$x,data_train$y)
        
                                                      z_train = apply(output_train,MARGIN=2,.self$Classify)
                                                      classification_error=base::sum(abs(z_train-data$bin_class))/length(data$bin_class) #starting value, to be optimized
                                                      
                                                      weight_gradient_layer <- matrix(nrow=layer$num_nodes_in,ncol=layer$num_nodes_out)
                                                      bias_gradient_layer <- rep(NA,layer$num_nodes_out)
                                                      for (i in 1:layer$num_nodes_in){
                                                        for (j in 1:layer$num_nodes_out){
                                                          #update weight (increment by stepwidth, approximate differential)
                                                          .self$layers[[layer_depth]]$weights[i,j]<-.self$layers[[layer_depth]]$weights[i,j]+stepwidth
                                                          
                                                          #calculate_updated output
                                                          output_train = mapply(.self$CalcOutputNN,data_train$x,data_train$y)
                                                          #classifiy updated Output
                                                          z_train = apply(output_train,MARGIN=2,.self$Classify)
                                                          #calculate new error
                                                          classification_error_new=base::sum(abs(z_train-data$bin_class))/length(data$bin_class) #starting value, to be optimized
                                      
                                                          #calculate direction of learning and stepwidth
                                                          delta_cost = classification_error_new - classification_error
                                                          #alter stepwidth to increase convergence speed
                                                          if (abs(delta_cost)>classification_error){
                                                            stepwidth = 2*stepwidth
                                                          }
                                                          else if (abs(delta_cost)<classification_error){
                                                            stepwidth = 0.5*stepwidth
                                                          }
                                                        
                                                          weight_gradient_layer[i,j] <- delta_cost/stepwidth
                                                          #reset weights
                                                          .self$layers[[layer_depth]]$weights[i,j]<-.self$layers[[layer_depth]]$weights[i,j]+stepwidth
                                                          
                                                        }
                                                      }
                                                      for (i in 1:layer$num_nodes_out){
                                                        
                                                        .self$layers[[layer_depth]]$bias[i] <- .self$layers[[layer_depth]]$bias[i]+stepwidth
                                                        
                                                        #calculate_updated output
                                                        output_train = mapply(.self$CalcOutputNN,data_train$x,data_train$y)
                                                        
                                                        z_train = apply(output_train,MARGIN=2,.self$Classify)
                                                        classification_error_new=base::sum(abs(z_train-data$bin_class))/length(data$bin_class) 
                                                        #calculate cost in comparison to starting point
                                                        delta_cost = classification_error_new - classification_error
                                                        bias_gradient_layer[i] <- delta_cost/stepwidth
                                                        .self$layers[[layer_depth]]$bias[i] <- .self$layers[[layer_depth]]$bias[i]-stepwidth #reset to starting position, i just wanted to screen sensitivity
                                                      }
                                                      .self$weight_gradient[[layer_depth]] <- weight_gradient_layer
                                                      .self$bias_gradient[[layer_depth]] <- bias_gradient_layer
                                                      print(.self$bias_gradient[[layer_depth]])
                                                      #perform learning step
                                                      .self$ApplyGradient(learnrate,layer_depth)
                                                      layer_depth = layer_depth+1
                                                    }
                                                    
                                                  },
                                                  
                                                  ApplyGradient = function(learnrate,layer_depth){
                                                    #for one specific layer I update the layer weights according to the gradients discovered in the learn step
                                                    layer <- .self$layers[[layer_depth]]
                                                    bias_tmp <- layer$bias
                                                    weights_tmp <- layer$weights
                                                    bias_gradient_tmp <- .self$bias_gradient[[layer_depth]]
                                                    weight_gradient_tmp <- .self$weight_gradient[[layer_depth]]
                                                    
                                                    
                                                    for (i in 1:.self$num_nodes_nn[[layer_depth]][2]){
                                                      bias_tmp[i] <- bias_tmp[i] - bias_gradient_tmp[i]*learnrate
                                                      for (j in 1:.self$num_nodes_nn[[layer_depth]][1]){
                                                        weights_tmp[j,i] <- weights_tmp[j,i] - weight_gradient_tmp[j,i]*learnrate
                                                      }
                                                    }
                                                    .self$layers[[layer_depth]]$weights <- weights_tmp
                                                    .self$layers[[layer_depth]]$bias <- bias_tmp
                                                    
                                                    
                                              
                                                  },
                                                  
                                                  random_weights_biases = function(layer_sizes,range_weights){
                                                    min_range = min(range_weights)
                                                    max_range = max(range_weights)
                                                    random_weights_mat = runif(n=prod(layer_sizes),min=min_range,max=max_range)
                                                    random_mat1 = matrix(head(random_weights_mat,(layer_sizes[1]*layer_sizes[2])),nrow=layer_sizes[1],ncol=layer_sizes[2])
                                                    random_mat2 = matrix(tail(random_weights_mat,(layer_sizes[2]*layer_sizes[3])),nrow=layer_sizes[2],ncol=layer_sizes[3])
                                                    random_mats = list(random_mat1,random_mat2)
                                                    
                                                    random_weights_biases=runif(n=layer_sizes[2]+layer_sizes[3],min=min_range,max=max_range)
                                                    random_bias_1 = head(random_weights_biases,layer_sizes[2])
                                                    random_bias_2 = tail(random_weights_biases,layer_sizes[3])
                                                    random_biases  =list(random_bias_1,random_bias_2)
                                                    
                                                    return(c(random_mats,random_biases))
                                                  },
                                                  
                                                  initialize_random_layers = function(...){
                                                    
                                                    random_init <- .self$random_weights_biases(layer_sizes=.self$layer_sizes,range_weights=c(-10,10))
                                                    
                                                    weight_list <- list(random_init[[1]],random_init[[2]])
                                                  
                                                    bias_list <- list(random_init[[3]],random_init[[4]])
                                                    
                                                    layer_1 <- layer$new(num_nodes_in=layer_sizes[1],num_nodes_out=layer_sizes[2],weights=weight_list[[1]],bias=bias_list[[1]])
                                                    layer_2 <- layer$new(num_nodes_in=layer_sizes[2],num_nodes_out=layer_sizes[3],weights=weight_list[[2]],bias=bias_list[[2]])
                                                    .self$layers <- list(layer_1,layer_2)
                                                      
                                                  },
                                                  ActivationFunction = function(output){ #sigmoid activation function, later i will also implement backpropagation analytically (shape of activation fct. will then become important)
                                                    return (sapply(output,FUN=function(x) 1/(1+exp(-x))))
                                                  }
                                                  
                                                  
                                                  
                                                ))



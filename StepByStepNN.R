
data <- data.frame(x=runif(1000),y=runif(1000)) %>% 
        mutate(label=ifelse(test=sqrt((x**2+y**2))<=0.5,yes = "red" ,no="blue")) 

data %>% ggplot(aes(x=x,y=y,colour=label))+geom_point()+theme_bw()

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




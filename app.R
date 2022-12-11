#
# This is a Shiny web application. You can run the application by clicking
# the 'Run App' button above.
#
# Find out more about building applications with Shiny here:
#
#    http://shiny.rstudio.com/
#

library(shiny)
library(ggplot2)
library(dplyr)
library(akima)
library(tidyr)


path=getwd()
source(paste0(path,"/StepByStepNN.R"))
consideredDigits <- 3
stepWidth <- 1/10^(consideredDigits+1)


# Define UI for application that draws a histogram
ui <- fluidPage(

    # Application title
    titlePanel("Simple DecisionBoundary Tool"),

    # Sidebar with a slider input for number of bins 
    headerPanel("Decision Boundary Params (One Hidden Layer with 3 Nodes)"),
    sidebarPanel(
      sliderInput(inputId = "W11", label = "W11", min = -1, max = 1, value = 0.1, step = stepWidth),
      sliderInput(inputId = "W12", label = "W12", min = -1, max = 1, value = 0.2, step = stepWidth),
      sliderInput(inputId = "W13", label = "W13", min = -1, max = 1, value = 0.3, step = stepWidth),
      sliderInput(inputId = "W21", label = "W21", min = -1, max = 1, value = 0.4, step = stepWidth),
      sliderInput(inputId = "W22", label = "W22", min = -1, max = 1, value = 0.5, step = stepWidth),
      sliderInput(inputId = "W23", label = "W23", min = -1, max = 1, value = 0.6, step = stepWidth),
      sliderInput(inputId = "bias1_1", label = "b1_1", min = -1, max = 1, value = 0.7, step = stepWidth),
      sliderInput(inputId = "bias2_1", label = "b2_1", min = -1, max = 1, value = 0.8, step = stepWidth),
      sliderInput(inputId = "bias3_1", label = "b3_1", min = -1, max = 1, value = 0.9, step = stepWidth),
      sliderInput(inputId = "A11", label = "A11", min = -1, max = 1, value = 0.2, step = stepWidth),
      sliderInput(inputId = "A21", label = "A21", min = -1, max = 1, value = 0.4, step = stepWidth),
      sliderInput(inputId = "A31", label = "A31", min = -1, max = 1, value = 0.5, step = stepWidth),
      sliderInput(inputId = "A12", label = "A12", min = -1, max = 1, value = 0.1, step = stepWidth),
      sliderInput(inputId = "A22", label = "A22", min = -1, max = 1, value = 0.7, step = stepWidth),
      sliderInput(inputId = "A32", label = "A32", min = -1, max = 1, value = 0.8, step = stepWidth),
      sliderInput(inputId = "bias1_2", label = "b1_2", min = -1, max = 1, value = 1, step = stepWidth),
      sliderInput(inputId = "bias2_2", label = "b2_2", min = -1, max = 1, value = 0, step = stepWidth)
    ),
    mainPanel(),

        # Show a plot of the generated distribution
        mainPanel(
           plotOutput("ClassifierPlot"),
           textOutput("selected_var")
           
        )
    
)

# Define server logic required to draw a histogram
server <- function(input, output) {
      
    
    data <- data.frame(x=runif(100),y=runif(100)) %>% 
      mutate(label=ifelse(test=sqrt((x**2+y**2))<=0.5,yes = "red" ,no="blue")) %>% mutate(bin_class = ifelse(label=="red",yes=1,no=0))
      
    output$ClassifierPlot <- renderPlot({
        
        layer_sizes = c(2,3,2)
        # generate bins based on input$bins from ui.R
        mat_1 = matrix(c(input$W11,input$W12,input$W13,input$W21,input$W22,input$W23),nrow=layer_sizes[1],ncol=layer_sizes[2])
        mat_2 = matrix(c(input$A11,input$A21,input$A31,input$A12,input$A22,input$A32),nrow=layer_sizes[2],ncol=layer_sizes[3])
        weights_list = list(mat_1,mat_2)
        bias_1 = c(input$bias1_1,input$bias2_1,input$bias3_1)
        bias_2 = c(input$bias1_2,input$bias2_2)
        biases = list(bias_1,bias_2)
        weights_list <- head(random_init,2)
        biases <- tail(random_init,2)
        
        nn <- NeuralNetwork$new(layer_sizes=layer_sizes,num_nodes_nn = list(c(2,3),c(3,2)),weights=weights_list,bias=biases)
        
        #randomize initial weights
        nn$initialize_random_layers()
        
        #TBA Later:Update displayed weights (sliders) according to learning process
        # random_init <- random_weights_biases(layer_sizes=layer_sizes,range_weights=c(-1,1))
        # nn$weights <- head(random_init,2)
        # nn$bias <- tail(random_init,2)
        # print(nn$weights)
        # 
        
        x=seq(from=0,to=1,length.out=10)
        y=x
        data_grid=expand.grid(x=x,y=y)
        
        #output=mapply(nn$CalcOutputNN,data_grid$x,data_grid$y,MoreArgs=list(layers=nn$GetLayers())) #to visualize decision boundary
        #z=apply(output,MARGIN=2,nn$Classify) #contour of classification of grid ("pixels of plot")
        
        #data_grid <- data_grid %>% mutate(z=z)
        

        output_train = mapply(nn$CalcOutputNN,data$x,data$y)
        z_train = apply(output_train,MARGIN=2,nn$Classify)
        classification_error=base::sum(abs(z_train-data$bin_class))/length(data$bin_class)
        print(classification_error)
        v <- ggplot(data=data, aes(x=x, y=y,colour=label))+
            geom_point()
        nn$Learn(data,0.25,0.0001) #testing
        for (i in 1:1000){
          nn$Learn(data,0.25,0.1)
          output_train = mapply(nn$CalcOutputNN,data$x,data$y)
          z_train = apply(output_train,MARGIN=2,nn$Classify)
          data_grid <- data_grid %>% mutate(z=z_train)
          classification_error=base::sum(abs(z_train-data$bin_class))/length(data$bin_class)
          #v<-v+geom_raster(data=data_grid,aes(x,y,fill=z_train),alpha=0.2)
          #print(v)
          print(paste("Average Error per Input:",classification_error))  
        }
        
    
        })
    
}

# Run the application 
shinyApp(ui = ui, server = server)

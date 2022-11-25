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
      sliderInput(inputId = "W11", label = "W11", min = -1, max = 1, value = 0.2, step = stepWidth),
      sliderInput(inputId = "W12", label = "W12", min = -1, max = 1, value = 0.2, step = stepWidth),
      sliderInput(inputId = "W13", label = "W13", min = -1, max = 1, value = 0.2, step = stepWidth),
      sliderInput(inputId = "W21", label = "W21", min = -1, max = 1, value = 0.2, step = stepWidth),
      sliderInput(inputId = "W22", label = "W22", min = -1, max = 1, value = 0.6, step = stepWidth),
      sliderInput(inputId = "W23", label = "W23", min = -1, max = 1, value = 0.6, step = stepWidth),
      sliderInput(inputId = "bias1_1", label = "b1_1", min = 0, max = 1, value = 0.6, step = stepWidth),
      sliderInput(inputId = "bias2_1", label = "b2_1", min = 0, max = 1, value = 0.6, step = stepWidth),
      sliderInput(inputId = "bias3_1", label = "b3_1", min = 0, max = 1, value = 0.6, step = stepWidth),
      sliderInput(inputId = "A11", label = "A11", min = -1, max = 1, value = 0.2, step = stepWidth),
      sliderInput(inputId = "A21", label = "A21", min = -1, max = 1, value = 0.2, step = stepWidth),
      sliderInput(inputId = "A31", label = "A31", min = -1, max = 1, value = 0.2, step = stepWidth),
      sliderInput(inputId = "A12", label = "A12", min = -1, max = 1, value = 0.2, step = stepWidth),
      sliderInput(inputId = "A22", label = "A22", min = -1, max = 1, value = 0.6, step = stepWidth),
      sliderInput(inputId = "A32", label = "A32", min = -1, max = 1, value = 0.6, step = stepWidth),
      sliderInput(inputId = "bias1_2", label = "b1_2", min = -1, max = 1, value = 1, step = stepWidth),
      sliderInput(inputId = "bias2_2", label = "b2_2", min = -1, max = 1, value = 0, step = stepWidth)
    ),
    mainPanel(),

        # Show a plot of the generated distribution
        mainPanel(
           plotOutput("ClassifierPlot")
           
        )
    
)

# Define server logic required to draw a histogram
server <- function(input, output) {
      
    
    data <- data.frame(x=runif(100),y=runif(100)) %>% 
      mutate(label=ifelse(test=sqrt((x**2+y**2))<=0.5,yes = "red" ,no="blue")) 
      
    output$ClassifierPlot <- renderPlot({
        
        # generate bins based on input$bins from ui.R
        mat_1 = matrix(c(input$W11,input$W12,input$W13,input$W21,input$W22,input$W23),nrow=2,ncol=3)
        mat_2 = matrix(c(input$A11,input$A21,input$A31,input$A12,input$A22,input$A32),nrow=3,ncol=2)
        weights_list = list(mat_1,mat_2)
        bias_1 = c(input$bias1_1,input$bias2_1)
        bias_2 = c(input$bias1_2,input$bias2_2)
        biases = list(bias_1,bias_2)
        layer_sizes = c(2,3,2)
        
        nn <- NeuralNetwork$new(layer_sizes=layer_sizes,num_nodes_out=2,weights=weights_list,bias=biases)
        
        x=seq(from=0,to=1,length.out=10)
        y=x
        data_grid=expand.grid(x=x,y=y)
        
        output=mapply(nn$CalcOutputNN,data_grid$x,data_grid$y) #to visualize decision boundary
  
        z=apply(output,MARGIN=2,nn$Classify)
  
        data_grid <- data_grid %>% mutate(z=z)
        
        v <- ggplot(data_grid, aes(x, y, fill= z)) + 
          geom_tile(alpha=0.2)
        v<-v + geom_point(data=data, aes(x=x, y=y,colour=label))
        print(v)
        
        
    })
}

# Run the application 
shinyApp(ui = ui, server = server)

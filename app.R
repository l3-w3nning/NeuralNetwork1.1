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

data <- data.frame(x=runif(3600),y=runif(3600)) %>% 
  mutate(label=ifelse(test=sqrt((x**2+y**2))<=0.5,yes = "red" ,no="blue")) 

# Define UI for application that draws a histogram
ui <- fluidPage(

    # Application title
    titlePanel("Simple DecisionBoundary Tool"),

    # Sidebar with a slider input for number of bins 
    headerPanel("Linear Decision Boundary Params"),
    sidebarPanel(
      sliderInput(inputId = "W11", label = "W11", min = -1, max = 1, value = 0.2, step = stepWidth),
      sliderInput(inputId = "W12", label = "W12", min = -1, max = 1, value = 0.2, step = stepWidth),
      sliderInput(inputId = "W21", label = "W21", min = -1, max = 1, value = 0.6, step = stepWidth),
      sliderInput(inputId = "W22", label = "W22", min = -1, max = 1, value = 0.6, step = stepWidth),
      sliderInput(inputId = "bias1", label = "b1", min = 0, max = 1, value = 0.6, step = stepWidth),
      sliderInput(inputId = "bias2", label = "b2", min = 0, max = 1, value = 0.6, step = stepWidth)
    ),
    mainPanel(),

        # Show a plot of the generated distribution
        mainPanel(
           plotOutput("ClassifierPlot")
           
        )
    
)

# Define server logic required to draw a histogram
server <- function(input, output) {
      
    reactive_fun <- reactive({get_weight_matrix_and_bias(as.numeric(input$W11),as.numeric(input$W12),as.numeric(input$W21),as.numeric(input$W22),as.numeric(input$b1),as.numeric(input$b2))})
      
      
    output$ClassifierPlot <- renderPlot({
        
        # generate bins based on input$bins from ui.R
        mat = matrix(c(input$W11,input$W12,input$W21,input$W22),nrow=2,ncol=2)
        bias = c(input$bias1,input$bias2)
        print(bias)
        print(mat)
        m <- (mat[2,2]-mat[1,2])/(mat[1,1]-mat[2,1])
        b <- bias[2]-bias[1]
        eq = function(x){m*x+b}
        
        x=seq(from=0,to=1,length.out=60)
        y=x
        data_grid=expand.grid(x=x,y=y)
        z=mapply(classifier,data_grid$x,data_grid$y,MoreArgs = list(mat,bias))
    
        data_grid <- data_grid %>% mutate(z=z)
        
        v <- ggplot(data_grid, aes(x, y, fill= z)) + 
          geom_tile(alpha=0.4)
        v<-v + geom_point(data=data, aes(x=x, y=y,colour=label))
        print(v)
        
        
    })
}

# Run the application 
shinyApp(ui = ui, server = server)

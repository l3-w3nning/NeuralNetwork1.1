library(shiny)
library(ggplot2)
library(dplyr)
library(shinythemes)
library(tidyr)
stepWidth=0.001

path=getwd()
source(paste0(path,"/StepByStepNN.R"))

linebreaks <- function(n){HTML(strrep(br(), n))}

ui<-shinyUI(
  navbarPage(
    theme = shinytheme("superhero"),
    title = "NeuralNetwork101",
    tabPanel("Schematic Sketch of Network",
             tags$figure(
               align = "center",
               tags$img(
                 src = "drawing.jpg",
                 width = 600,
                 alt = "If output1>output2 classify as red, otherwise blue"
               ),
               tags$figcaption("If output1>output2 classify as red, otherwise blue, i also showcased the calculation of the hidden layer (chained with the activation function sigma)")
             )),
    tabPanel(
      "Manual Parameter Input",
      mainPanel(
        fluidRow(
          #column(6,img(src="drawing.jpg",width=400)),
          column(12,
                 
                 plotOutput('ClassifierPlot'),
          )  
        )
        
      ),
      
      
      fluidRow(
        headerPanel("Please set the parameters yourself!"),
        br(),
        column(3,
               h4("Weights First Layer"),
               sliderInput(inputId = "W11", label = "W11", min = -10, max = 10, value = 5, step = stepWidth),
               sliderInput(inputId = "W12", label = "W12", min = -10, max = 10, value = 5, step = stepWidth),
               sliderInput(inputId = "W13", label = "W13", min = -10, max = 10, value = 6, step = stepWidth),
               sliderInput(inputId = "W21", label = "W21", min = -10, max = 10, value = 7, step = stepWidth),
               sliderInput(inputId = "W22", label = "W22", min = -10, max = 10, value = -2, step = stepWidth),
               sliderInput(inputId = "W23", label = "W23", min = -10, max = 10, value = 3, step = stepWidth)
        ),
        column(3,
               h4("Bias First Layer"),
               sliderInput(inputId = "bias1_1", label = "b1_1", min = -10, max = 10, value = 7, step = stepWidth),
               sliderInput(inputId = "bias2_1", label = "b2_1", min = -10, max = 10, value = 8, step = stepWidth),
               sliderInput(inputId = "bias3_1", label = "b3_1", min = -10, max = 10, value = 9, step = stepWidth)
        ),
        column(3,
               h4("Weights Second Layer"),
               sliderInput(inputId = "A11", label = "A11", min = -10, max = 10, value = 2, step = stepWidth),
               sliderInput(inputId = "A21", label = "A21", min = -10, max = 10, value = 4, step = stepWidth),
               sliderInput(inputId = "A31", label = "A31", min = -10, max = 10, value = 5, step = stepWidth),
               sliderInput(inputId = "A12", label = "A12", min = -10, max = 10, value = 1, step = stepWidth),
               sliderInput(inputId = "A22", label = "A22", min = -10, max = 10, value = 7, step = stepWidth),
               sliderInput(inputId = "A32", label = "A32", min = -10, max = 10, value = 8, step = stepWidth)
        ),
        column(3,
               h4("Bias Second Layer"),
               sliderInput(inputId = "bias1_2", label = "b1_2", min = -10, max = 10, value = 8, step = stepWidth),
               sliderInput(inputId = "bias2_2", label = "b2_2", min = -10, max = 10, value = 0, step = stepWidth)
        )
      )
      ),
    tabPanel(
      "Learn Optimal Parameters",
      sidebarPanel(
        sliderInput(inputId = "learnrate", label = "learnrate", min = 0, max = 0.5, value = 0.25, step = stepWidth),
        sliderInput(inputId = "stepwidth", label = "stepwidth", min = 0, max = 0.01, value = 0.001, step = stepWidth),
        sliderInput(inputId = "epochs", label = "epochs", min = 10, max = 1000, value = 100, step = 1),
        br(),
        actionButton("LearnButton2", "Learn More!"),
        p("Click the button to let the Neural Network Learn n times (n=epochs)"),
      ),
      mainPanel(
        br(),
        br(),
        br(),
        plotOutput('ClassifierPlotLearned'),
        br(),
        tableOutput("learnmore"),
      ),
    )
  )
)

server <- function(input, output) {
  
  
  data_train <- data.frame(x=runif(400),y=runif(400)) %>% 
    mutate(label=ifelse(test=sqrt((x**2+y**2))<=0.5,yes = "red" ,no="blue")) %>% mutate(bin_class = ifelse(label=="red",yes=1,no=0))
  
  autoInvalidate <- reactiveTimer(1000) #update plots every second
  
  output$ClassifierPlot <- renderPlot({
    
    layer_sizes = c(2,3,2)
    # generate bins based on input$bins from ui.R
    mat_1 = matrix(c(input$W11,input$W12,input$W13,input$W21,input$W22,input$W23),nrow=layer_sizes[1],ncol=layer_sizes[2])
    mat_2 = matrix(c(input$A11,input$A21,input$A31,input$A12,input$A22,input$A32),nrow=layer_sizes[2],ncol=layer_sizes[3])
    weights_list = list(mat_1,mat_2)
    bias_1 = c(input$bias1_1,input$bias2_1,input$bias3_1)
    bias_2 = c(input$bias1_2,input$bias2_2)
    biases = list(bias_1,bias_2)
    
    nn <- NeuralNetwork$new(layer_sizes=layer_sizes,num_nodes_nn = list(c(2,3),c(3,2)),weights=weights_list,bias=biases)
    nn$initialize_random_layers()
    #ich muss erst random intiialisieren, damit layers eigenschaft entsteht und classified werden kann warum?
    x=seq(from=0,to=1,length.out=20)
    y=x
    data_grid=expand.grid(x=x,y=y)
    
    
    output_train = mapply(nn$CalcOutputNN,data_train$x,data_train$y)
    z_train = apply(output_train,MARGIN=2,nn$Classify)
    colors = c("blue","red")
    names(colors) = levels(factor(levels(z_train)))
    colorscale = scale_fill_manual(name=z_train,values=colors)
    
    classification_error=base::sum(abs(z_train-data_train$bin_class))/length(data_train$bin_class)
    print(classification_error)
    v <- ggplot()+
      geom_point(data=data_train, aes(x=x, y=y,colour=label))+theme(legend.position = "none")
    v <- v +
      geom_raster(data= data_grid, aes(x,y,fill = z_train), interpolate = TRUE,alpha=0.2)+
      colorscale+
      labs(title=paste("Correcly classified points with current setup:",100*classification_error,"%"))+
      xlab("X-Data")+ylab("Y-Data")+
      scale_fill_gradient(low="blue",high="red")
    print(v)
     
    
  })
  
    observeEvent(input$LearnButton2, {
      layer_sizes = c(2,3,2)
      #get new nn object
      nn <- NeuralNetwork$new(layer_sizes=layer_sizes,num_nodes_nn = list(c(2,3),c(3,2)))
      
    
      #for first click on learnbutton 2 initialize nn with random weights and biases and calculate original classification error
      if (input$LearnButton2==1){#when nn is initialized randomly, calculate base classification error
        nn$initialize_random_layers() 
        output_train = mapply(nn$CalcOutputNN,data_train$x,data_train$y)
        z_train = apply(output_train,MARGIN=2,nn$Classify)
        classification_error_orig=base::sum(abs(z_train-data_train$bin_class))/length(data_train$bin_class)
      } 
     
      
      x=seq(from=0,to=1,length.out=20)
      y=x
      data_grid=expand.grid(x=x,y=y)
      
      
      v <- ggplot()+
        geom_point(data=data_train, aes(x=x, y=y,colour=label))+theme(legend.position = "none")
      v <- v +
        geom_raster(data= data_grid, aes(x,y,fill = z_train), interpolate = TRUE,alpha=0.2)+
        labs(title=paste("Correcly classified points with current setup:",100*classification_error,"%"))+
        xlab("X-Data")+ylab("Y-Data")+
        scale_fill_gradient(low="blue",high="red")
      output$ClassifierPlotLearned <- renderPlot({
        v
        })
    learnrate = input$learnrate
    stepwidth = input$stepwidth
    epochs = input$epochs
    
    for (i in 1:epochs){
      nn$Learn(data_train,learnrate,stepwidth)
      output_train = mapply(nn$CalcOutputNN,data_train$x,data_train$y)
      z_train = apply(output_train,MARGIN=2,nn$Classify)
      data_grid <- data_grid %>% mutate(z=z_train)
      classification_error=base::sum(abs(z_train-data_train$bin_class))/length(data_train$bin_class)
      print(classification_error)
      display_data = data.frame("ClassificationErrorOrig"=classification_error_orig,"ClassificationErrorNew"=classification_error)
      plot <-ggplot()+
        geom_point(data=data_train, aes(x=x, y=y,colour=label))+theme(legend.position = "none")+geom_raster(data=data_grid,aes(x,y,fill=z_train),interpolate=TRUE,alpha=0.2)
      output$learnmore <- renderTable(display_data)
      output$ClassifierPlotLearned <- renderPlot({
        autoInvalidate()
        plot
      })
      
    }
     
      })
    
    
    
  
  
}
    

shinyApp(ui,server)
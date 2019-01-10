# Inteligencia de negocio 2018/2019
# Solución 3
# Autor: Luis Balderas Ruiz


library(xgboost)
library(Matrix)
library(MatrixModels)
library(datos.table)


test<-read.csv('/home/luisbalru/Universidad/Business-Intelligence/P3/data/water_pump_tst.csv')


test$status_group <- 0


train<-read.csv('/home/luisbalru/Universidad/Business-Intelligence/P3/data/water_pump_tra.csv')
etiqueta<-read.csv('/home/luisbalru/Universidad/Business-Intelligence/P3/data/water_pump_tra_target.csv')


etiqueta <- subset(etiqueta, select = status_group )


train<-cbind(train,etiqueta)


train$status_group<-0

# Diferenciando en test y train
train$tst <- 0
test$tst <- 1

# Combinando test y train
datos<- rbind(train,test)


datos$date_recorded<-as.Date(datos$date_recorded)


datos$region_code<-factor(datos$region_code)
datos$district_code<-factor(datos$district_code)


min_year<-1960
datos$construction_year<-datos$construction_year-min_year


datos$construction_year[datos$construction_year<0]= median(datos$construction_year[datos$construction_year>0])


datos$gps_height[datos$gps_height==0]=median(datos$gps_height[datos$gps_height>0])



# SELECCIÓN DE CARACTERÍSTICAS

datos$num_private<-NULL


datos$recorded_by<-NULL


datos$wpt_name<-NULL


datos$extraction_type_group<-NULL
datos$extraction_type<-NULL


datos$payment_type<-NULL


datos$water_quality<-NULL


datos$scheme_management<-NULL


datos$district_code<-NULL
datos$region<-NULL
datos$region_code<-NULL
datos$subvillage<-NULL
datos$ward<- NULL


datos$waterpoint_type_group<-NULL


datos$quantity_group<-NULL


datos$installer<-NULL

# Separando de nuevo para aplicar los modelos
datos_train <- datos[datos$tst==0,]
datos_test <- datos[datos$tst==1,]

# Elimino id de test para que test y train tengan el mismo número de columnas
datos_test.noID<-subset(datos_test, select = -id)
datos_test.noID$status_group <- NULL

#Elimino id y status_group de train
datos_train<-subset(datos_train, select = c(-id,-status_group))


datos_test.noID <- as.matrix(as.datos.frame(lapply(datos_test.noID, as.numeric)))
datos_train <- as.matrix(as.datos.frame(lapply(datos_train, as.numeric)))
etiqueta<-as.numeric(etiqueta$status_group)

#xgb.DMatrix para usar el modelo xgboost
train.DMatrix <- xgb.DMatrix(datos = datos_train,etiqueta = etiqueta, missing = NA)


# Bucle con 11 iteraciones (se ha modificado en mejora de la implementación) con diferentes semillas para mejorar el modelo


i=2

solucion.table<-datos.frame(id=datos_test[,"id"])

for (i in 2:25){
  set.seed(i)

  # Validación cruzada para determinar cuántas veces ejecuto el modelo
  xgb.tab = xgb.cv(data = train.DMatrix, objective = "multi:softmax", booster = "gbtree",
                   nrounds = 500, nfold = 4, early.stop.round = 10, num_class = 4, maximize = FALSE,
                   evaluation = "merror", eta = 0.35, max_depth = 14, colsample_bytree = 0.4)
  # Variable para identificar el número de iteraciones calculado previamente
  min.error.idx = which.min(xgb.tab$evaluation_log[, test_merror_mean])
  
  #Modelo
  model <- xgboost(data = train.DMatrix, objective = "multi:softmax", booster = "gbtree",
                   eval_metric = "merror", nrounds = min.error.idx, 
                   num_class = 4,eta = 0.35, max_depth = 14, colsample_bytree = 0.4)
  

  
  predict <- predict(model,datos_test.noID)
  
  predict[predict==1]<-"functional"
  predict[predict==2]<-"functional needs repair"
  predict[predict==3]<-"non functional"
  
  
  solucion.table[,i]<-predict
}


solucion.table.count<-apply(solution.table,MARGIN=1,table)

predict.combined<-vector()

x=1
for (x in 1:nrow(datos_test)){
  predict.combined[x]<-names(which.max(solution.table.count[[x]]))}


table(predict.combined)


solucion<- datos.frame(id=datos_test[,"id"], status_group=predict.combined)


#Submission
write.csv(solucion, file = "submission.csv", row.names = FALSE)

#Importancia de cada variable en el modelo
importancia <- xgb.importance(feature_names = colnames(datos_train), model =model)
importancia
xgb.plot.importance(importance_matrix = importancia)


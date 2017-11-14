#Подгружаем необходимые библиотеки
library(caret)
library(lime)
library(dplyr)

#Работаем с датасетом housing, который заранее разделили на тестовую и тренировочную выборки
#Загружаем тестовую, тренировочную выборки и датасет с целевой переменной для тренировочной выборки
test <- read_csv("/students/oyasilyutina/test.csv")
train <- read_csv("/students/oyasilyutina/train.csv")
target <- read_csv("/students/oyasilyutina/target.csv")

#Создаем вектор с целевой переменной
target <- target[[1]]

#Переводим данные в формат data.frame
test <- as.data.frame(test)
train <- as.data.frame(train)

#Прогоняем модель линейной регрессии
model <- train(train, target, method = 'lm')

pred <- predict(test, model)

#Создаем объект для объянения работы модели
explainer <- lime(train, model)
class(explainer)

#Объясняем наблюдения из тестовой выборки
explanation <- explain(test[26,], explainer, n_features = 3)

#Визуализируем получившееся объяснение
plot_features(explanation)
summary(model)
print(explanation)

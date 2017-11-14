#Подгружаем все необходимые пакеты
library(text2vec)
library(xgboost)
library(readr)
library(lime)

#Загружаем данные с тестовой и тренировочной выборкой
X_test <- read_csv("/students/oyasilyutina/LIME_tutorial/X_test.csv")
X_train <- read_csv("/students/oyasilyutina/LIME_tutorial/X_train.csv")

#Приводим целевую переменную к классу character
X_test$type <- as.character(X_test$type)
X_train$type <- as.character(X_train$type)

#Приводим датасеты к формату data.frame
X_test <- as.data.frame(X_test)
X_train <- as.data.frame(X_train)

#Прописываем функцию для токенизаци
get_matrix <- function(text) {
  it <- itoken(text, progressbar = FALSE)
  create_dtm(it, vectorizer = hash_vectorizer())
}

#Применяем функцию get_matrix(), которая позволяет создать матрицу с документами и словами
dtm_train = get_matrix(X_train$questions)

#Прогоняем модель, которая будет классифицировать тексты вопросов на две группы: 1 - те вопросы, на
#которые доктор прописал лекарства от Сердечно-сосудистых заболеваний, 0 - те жалобы, на которые
#доктор прописал любой другой тип лекарств
xgb_model <- xgb.train(list(max_depth = 7, eta = 0.1, objective = "binary:logistic",
                            eval_metric = "error", nthread = 1),
                       xgb.DMatrix(dtm_train, label = X_train$type == "C"),
                       nrounds = 50)

#Выбираем текст вопроса из тестовой выборки, на котором будет продемонстрирована работа модели
sentences <- (X_test[145, 1])
sentences

#Применяем функцию lime(), которая, в свою очередь, создает другую функцию explainer
#Она используется в дальнейшем в качестве аргумента для объяснения работы модели
#Здесь также применяется функция для токенизации в качестве аргумента, так как тестовую выборку
#также необходимо превратить в матрицу документ-слово.
explainer <- lime(X_train$question, xgb_model, get_matrix)

#Для выбранного предложения используем полученый выше аргумент explainer, указываем количество 
#лейблов для рассматриваемого кейса, указываем количество предикторов, которые будут отражаться на 
#графике (в данном случае это слова)
explanations <- explain(sentences, explainer, n_labels=1, n_features = 4)

#Результат выполнения данных строк показывает общую точность модели. Она составляет 0.5402228
TEST_SMM <- sparse.model.matrix(type ~ ., data = X_test)
PRED <- predict(xgb_model, TEST_SMM)
max(PRED)

print(explanations)

#Визуализируем предсказания модели на выбранном кейсе
plot_text_explanations(explanations)

#Используем существующую модель для предсказания текста, который прописывается сразу в Shiny 
interactive_text_explanations(explainer)

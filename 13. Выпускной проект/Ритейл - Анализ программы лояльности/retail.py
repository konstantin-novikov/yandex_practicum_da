#!/usr/bin/env python
# coding: utf-8

# ## Материалы 
# 
# *  Дашборд https://public.tableau.com/app/profile/nvk2023.nbk2024/viz/dash_with_project/Dashboard1?publish=yes
# *  Презентация  https://disk.yandex.ru/i/-rBTfVvDElzVvA

# <h1>Финальный проект. Ритейл - Анализ программы лояльности<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Материалы" data-toc-modified-id="Материалы-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Материалы</a></span></li><li><span><a href="#Выгрузка-и-подготовка-данных-к-анализу" data-toc-modified-id="Выгрузка-и-подготовка-данных-к-анализу-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Выгрузка и подготовка данных к анализу</a></span><ul class="toc-item"><li><span><a href="#Определяем-наличие-пропусков-и-формат-данных" data-toc-modified-id="Определяем-наличие-пропусков-и-формат-данных-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>Определяем наличие пропусков и формат данных</a></span></li><li><span><a href="#Проверка-на-наличия-дубликатов" data-toc-modified-id="Проверка-на-наличия-дубликатов-2.2"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>Проверка на наличия дубликатов</a></span><ul class="toc-item"><li><span><a href="#Дубли-в-retail" data-toc-modified-id="Дубли-в-retail-2.2.1"><span class="toc-item-num">2.2.1&nbsp;&nbsp;</span>Дубли в retail</a></span></li><li><span><a href="#Дубли-в-product" data-toc-modified-id="Дубли-в-product-2.2.2"><span class="toc-item-num">2.2.2&nbsp;&nbsp;</span>Дубли в product</a></span></li></ul></li><li><span><a href="#Проверка-на-наличие-аномальных-значений" data-toc-modified-id="Проверка-на-наличие-аномальных-значений-2.3"><span class="toc-item-num">2.3&nbsp;&nbsp;</span>Проверка на наличие аномальных значений</a></span><ul class="toc-item"><li><span><a href="#Аномалии-в-датасете-retail" data-toc-modified-id="Аномалии-в-датасете-retail-2.3.1"><span class="toc-item-num">2.3.1&nbsp;&nbsp;</span>Аномалии в датасете retail</a></span></li><li><span><a href="#Аномалии-в-датасете-product" data-toc-modified-id="Аномалии-в-датасете-product-2.3.2"><span class="toc-item-num">2.3.2&nbsp;&nbsp;</span>Аномалии в датасете product</a></span></li></ul></li><li><span><a href="#Объединение-в-одну-таблицу" data-toc-modified-id="Объединение-в-одну-таблицу-2.4"><span class="toc-item-num">2.4&nbsp;&nbsp;</span>Объединение в одну таблицу</a></span></li><li><span><a href="#Итоги" data-toc-modified-id="Итоги-2.5"><span class="toc-item-num">2.5&nbsp;&nbsp;</span>Итоги</a></span></li></ul></li><li><span><a href="#Исследовательский-анализ-изученения-эффективности-программы-лояльности" data-toc-modified-id="Исследовательский-анализ-изученения-эффективности-программы-лояльности-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Исследовательский анализ изученения эффективности программы лояльности</a></span><ul class="toc-item"><li><span><a href="#Какую-прибыль-принесли-авторизованные-клиенты-и-какую-анонимные?" data-toc-modified-id="Какую-прибыль-принесли-авторизованные-клиенты-и-какую-анонимные?-3.1"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>Какую прибыль принесли авторизованные клиенты и какую анонимные?</a></span></li><li><span><a href="#Сколько-всего-клиентов-и-датасете?" data-toc-modified-id="Сколько-всего-клиентов-и-датасете?-3.2"><span class="toc-item-num">3.2&nbsp;&nbsp;</span>Сколько всего клиентов и датасете?</a></span></li><li><span><a href="#Сколько-клиентов-входят-в-программу-лояльности?" data-toc-modified-id="Сколько-клиентов-входят-в-программу-лояльности?-3.3"><span class="toc-item-num">3.3&nbsp;&nbsp;</span>Сколько клиентов входят в программу лояльности?</a></span></li><li><span><a href="#Как-часто-совершаются-покупки-в-разных-группах?" data-toc-modified-id="Как-часто-совершаются-покупки-в-разных-группах?-3.4"><span class="toc-item-num">3.4&nbsp;&nbsp;</span>Как часто совершаются покупки в разных группах?</a></span></li><li><span><a href="#Какова-разница-в-среднем-чеке-между-группами?" data-toc-modified-id="Какова-разница-в-среднем-чеке-между-группами?-3.5"><span class="toc-item-num">3.5&nbsp;&nbsp;</span>Какова разница в среднем чеке между группами?</a></span></li><li><span><a href="#Есть-ли-зависмость-вежду-количество-товаров-в-чеке-и-вхождение-в-группу-лояльности" data-toc-modified-id="Есть-ли-зависмость-вежду-количество-товаров-в-чеке-и-вхождение-в-группу-лояльности-3.6"><span class="toc-item-num">3.6&nbsp;&nbsp;</span>Есть ли зависмость вежду количество товаров в чеке и вхождение в группу лояльности</a></span></li><li><span><a href="#Сколько-всего-магазинов--в-данных-?" data-toc-modified-id="Сколько-всего-магазинов--в-данных-?-3.7"><span class="toc-item-num">3.7&nbsp;&nbsp;</span>Сколько всего магазинов  в данных ?</a></span></li><li><span><a href="#Какие-магазины-участвовали-в-данной-программе?" data-toc-modified-id="Какие-магазины-участвовали-в-данной-программе?-3.8"><span class="toc-item-num">3.8&nbsp;&nbsp;</span>Какие магазины участвовали в данной программе?</a></span></li><li><span><a href="#Какие-показатели-продаж-в-разных-магазинах?" data-toc-modified-id="Какие-показатели-продаж-в-разных-магазинах?-3.9"><span class="toc-item-num">3.9&nbsp;&nbsp;</span>Какие показатели продаж в разных магазинах?</a></span></li><li><span><a href="#Самые-преданные-покупатели,-сколько-раз-они-совершили-покупки?" data-toc-modified-id="Самые-преданные-покупатели,-сколько-раз-они-совершили-покупки?-3.10"><span class="toc-item-num">3.10&nbsp;&nbsp;</span>Самые преданные покупатели, сколько раз они совершили покупки?</a></span></li><li><span><a href="#Какие-товары--самые-популярные?" data-toc-modified-id="Какие-товары--самые-популярные?-3.11"><span class="toc-item-num">3.11&nbsp;&nbsp;</span>Какие товары  самые популярные?</a></span></li><li><span><a href="#Расчет-значения-LTV" data-toc-modified-id="Расчет-значения-LTV-3.12"><span class="toc-item-num">3.12&nbsp;&nbsp;</span>Расчет значения LTV</a></span></li></ul></li><li><span><a href="#Проверка-статистических-гипотез" data-toc-modified-id="Проверка-статистических-гипотез-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Проверка статистических гипотез</a></span><ul class="toc-item"><li><span><a href="#Посчитаем-95-й-и-99-й-перцентили-стоимости-товара" data-toc-modified-id="Посчитаем-95-й-и-99-й-перцентили-стоимости-товара-4.1"><span class="toc-item-num">4.1&nbsp;&nbsp;</span>Посчитаем 95-й и 99-й перцентили стоимости товара</a></span></li><li><span><a href="#Посчитаем-95-й-и-99-й-перцентили-количества-товаров-в-чеке" data-toc-modified-id="Посчитаем-95-й-и-99-й-перцентили-количества-товаров-в-чеке-4.2"><span class="toc-item-num">4.2&nbsp;&nbsp;</span>Посчитаем 95-й и 99-й перцентили количества товаров в чеке</a></span></li><li><span><a href="#Готовим-группы" data-toc-modified-id="Готовим-группы-4.3"><span class="toc-item-num">4.3&nbsp;&nbsp;</span>Готовим группы</a></span></li><li><span><a href="#Как-распределены-выборки-по-сумме-чека?" data-toc-modified-id="Как-распределены-выборки-по-сумме-чека?-4.4"><span class="toc-item-num">4.4&nbsp;&nbsp;</span>Как распределены выборки по сумме чека?</a></span></li><li><span><a href="#Как-распределены-выборки-по-количеству-товаров-в-чеке?" data-toc-modified-id="Как-распределены-выборки-по-количеству-товаров-в-чеке?-4.5"><span class="toc-item-num">4.5&nbsp;&nbsp;</span>Как распределены выборки по количеству товаров в чеке?</a></span></li><li><span><a href="#Средний-чек-участников-программы-лояльности-выше,-чем-у-остальных-покупателей." data-toc-modified-id="Средний-чек-участников-программы-лояльности-выше,-чем-у-остальных-покупателей.-4.6"><span class="toc-item-num">4.6&nbsp;&nbsp;</span>Средний чек участников программы лояльности выше, чем у остальных покупателей.</a></span></li><li><span><a href="#Среднее-количесто-товаров-в-чеке-у-участников-программы-лояльности-выше,-чем-у-остальных-покупателей" data-toc-modified-id="Среднее-количесто-товаров-в-чеке-у-участников-программы-лояльности-выше,-чем-у-остальных-покупателей-4.7"><span class="toc-item-num">4.7&nbsp;&nbsp;</span>Среднее количесто товаров в чеке у участников программы лояльности выше, чем у остальных покупателей</a></span></li></ul></li><li><span><a href="#Итоги" data-toc-modified-id="Итоги-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Итоги</a></span></li></ul></div>

# # Описание данных:
# 
# Датасет содержит данные о покупках в магазине строительных материалов «Строили, строили и наконец построили». Все покупатели могут получить доступ в магазин с использованием персональных карт. За 200 рублей в месяц они могут стать участниками программы лояльности. В программу включены скидки, специальные предложения, подарки.
# 
# Файл retail_dataset.csv:
# 
# - `purchaseId` — id чека;
# - `item_ID` — id товара;
# - `purchasedate` — дата покупки;
# - `Quantity` — количество товара;
# - `CustomerID` — id покупателя;
# - `ShopID` — id магазина;
# - `loyalty_program` — участвует ли покупатель в программе лояльности;
# 
# Файл product_codes.csv:
# 
# - `productID` — id товара;
# - `price_per_one` — стоимость одной единицы товара;
# 
# # Декомпозиция
# 
# 1️⃣ Выгрузка данных юпитер и первичное знакомство с данными. Обработка форматов, названий колонок, анализ пропусков и дубликатов, а так же принятие решение о внесении корректирующих изменений.
# 
# 2️⃣ Исследовательский анализ включающий себя изученение эффективности программы лояльности. Для этого сравним различные показатели между обычными покупателями и покупателями с привелегиями, а именно их соотношение, регулярность покупок, средний чек, зависимость от количества товаров в одном чеке между группами, а так же разницу между показтелями разных филиалов сети. Заодно посмторим на самые популярные товары и как их продажи распределены между друг другом, их завимость от программы лоялности и от временного периода.
# 
# 3️⃣ Сформулируем и проверим гипотезы
# 
# * Средний чек участников программы лояльности выше, чем у остальных покупателей.
# * Среднее количестов товаров в чек у участников программы лояльности выше, чем у остальных покупателей.
# 
# 4️⃣ На основании полученнхы данных сделаем выводы и напишем рекомендации, стоит ли продолжать данную программу или нет. Возможно, предложим свои варианты видения жизнеспособоности в зависимости от результатов.
# 
# 5️⃣ Данные для коллег оформим в юпитере и табло. Для заказчика в виде презентации и дашборда.
# 

# ## Выгрузка и подготовка данных к анализу

# In[1]:


# импортирую необходимые библиотеки
import pandas as pd
import scipy.stats as stats
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import math as mth
import re

get_ipython().run_line_magic('matplotlib', 'inline')

import plotly.express as px
from plotly import graph_objects as go

from scipy import stats as st

from datetime import datetime, timedelta

pd.set_option('display.max_colwidth', 0)


# In[2]:


#указываю путь к датасетам
try:
    retail_data = pd.read_csv('/datasets/retail_dataset.csv')
    product_data = pd.read_csv('/datasets/product_codes.csv')
    
except:
    retail_data = pd.read_csv('retail_dataset.csv')
    product_data = pd.read_csv('product_codes.csv')
    
display(retail_data, product_data)    


# ### Определяем наличие пропусков и формат данных

# In[3]:


retail_data.info() 


# В данных о чеках 105 335 строк, пропуски есть лишь в одной колонке CustomerID. Это объясняется тем, что покупатель не предьявил карту при оплате, поэтому его не идентифицировать. Количестов таких строк очень большое, поэтому надо учитывать этот момент при формирование выводов. 
# 
# Что касается формата данных. Такие колонки  loyalty_program приведу в булевый формат, а  purchasedate во временной.
# 
# Помимом этого, для удобства, приведу название колонок к нижнему регистру и проведу переименования некоторых из них

# In[4]:


#нижний регситр через анонимную функцию 
retail_data.columns = [x.lower() for x in retail_data.columns]

#переименуем через rename
retail = retail_data.rename(columns = {'purchaseid':'purchase_id', 'purchasedate':'date', 'customerid' : 'customer_id',                                   'shopid':'shop_id', 'loyalty_program':'loyalty'})

#меняем формат данных
retail['loyalty']=retail['loyalty'].astype('bool')
retail['date']=pd.to_datetime(retail['date'], format='%Y-%m-%d %H:%M:%S').dt.floor('D')


#пропуски в колонке customer_id заменю на ноль и так же поменяют формат
retail['customer_id']=retail['customer_id'].fillna(0).astype(int)


# А сейчас очередь второго датасета

# In[5]:


product_data.info() 


# Пропусков нет, формат данных соответсвует действительности

# ### Проверка на наличия дубликатов
# #### Дубли в retail

# In[6]:


#запускаем проверку
retail.duplicated().value_counts()


# На выходе получаем 1033 дубликата,посмотри,но что они из себя представляют

# In[7]:


display(retail[retail.duplicated() == True].sort_values(by ='purchase_id',ascending=False).head(20),         retail[retail.duplicated() == True].sort_values(by ='purchase_id',ascending=True).head(20))


# Настораживает колонка quantity, в которой есть и нулевые значения, и даже отрицательные. Возможно, это связано из-за возврата товара или корректировки позиций.
# В любом случае эти проблемные данны состявлять менее 1% от общего датафрейма и от них лучше избавиться. 

# In[8]:


retail = retail.drop_duplicates().reset_index(drop=True)


# #### Дубли в product
# Приступаем к проверке второго датасета

# In[9]:


product_data['productID'].value_counts().reset_index().query('productID >1')


# Почти в 2.5 тыщ артикулах у нас несколько цен. Понятно, что цены с течением времени меняются, но т.к. у нас нет данных о времени их изменения, то для будущих расчетов потребуется использовать их медианное значение.

# In[10]:


#но перед этим уберем нулевые значения стоимости
product = product_data.query('price_per_one != 0')

#далее сгруппируем данные по id и медианному значению
product = product.groupby('productID').agg({'price_per_one': 'median'}).reset_index()


# ### Проверка на наличие аномальных значений
# #### Аномалии в датасете retail
# В данном пункет посмотрим на значение в столбцах с количество, цено и номером магазина

# In[11]:


retail['quantity'].plot(kind='box', title='Диаграмма размаха количества')
plt.show()


# Во первых количество не может быть отрицательным, а во вторых очень 80 000 это явная аномалия. Посмотрим на них поближе

# In[12]:


retail.query('quantity<=0').sort_values(by ='quantity', ascending=False)


# Такие данные нельзя использовать в товарообороте и вести складской учет. В добавок на данные момент нас интересует успех программы лоялности, которая по задумке, должна поднимать продажи и даже если это окажутся какие-либо возвраты, то оносится они будут к другому исследованию. Поэтому придется он них избавиться. Но чтобы понять какой диапазон использовать, воспользуемся специальным методом

# In[13]:


retail['quantity'].quantile([0.95])


# In[14]:


#удаляем данные меньше нуля и аномальное значение свыше 10 000
retail = retail.query('24> quantity > 0')

#заново строим диаграмму
retail['quantity'].plot(kind='box', title='Диаграмма размаха количества')


# Посмотрим на список магазинов 

# In[15]:


retail.shop_id.value_counts().reset_index().head()


# Shop 0 аномально большое количество покупок. Рискну предположить, что это интернет-магазин, с возможность забора заказа в любом филиале. Или же какой-то центральный магазин в городе миллионнике, а остальные это филиалы по стране. 

# #### Аномалии в датасете product
# Проверим значения цен 

# In[16]:


product['price_per_one'].plot(kind='box', title='Диаграмма размах цены')
plt.show()


# Наблюдается колосальный еденичный выброс, который может исказить картину анализа. Поэтому убираю из анализа цены выше 1000

# In[17]:


#удаляем аномальное значение свыше 1000
product = product.query('price_per_one < 1000')


# In[18]:


print(f'После устранение пропусков, дублей и аномалий от перовочального датасета retail осталось {round(retail.shape[0]/retail_data.shape[0]*100)}%, а данные о ценах усреднились и количество строк в product стало меньше на {100-round(product.shape[0]/product_data.shape[0]*100)}% ')


# ### Объединение в одну таблицу
# Это необходимо, чтобы соотнести id товара с логов кассовых аппаратов с id стоимости товаров из второго файла. Благодаря этому у нас появиться возможность провести расчеты по среднему чеку

# In[19]:


data = pd.merge(retail, product, how='left', on=None, left_on='item_id', right_on='productID') 
data.info()


# In[20]:


#Из-за удаления цены свыше 1000, в новой таблице потярлись значниея 8 строк, удалим их
data=data.dropna()

#добавим колонку с суммой
data['total']=data['quantity']*data['price_per_one']
data


# ### Итоги
# На начальном этапе исследования в нашем распоряжение был датафрейм с логами от кассовых аппаратов и с данными о стоимости товаров. В результате “очищения” данных были приняты следующее решения:
# 
# 
# *	Пропуски в колонке идентификатора пользователя были заменены на 0
# 
# 
# *	Дубликаты в стоимости 1 и того же id товара заменили медианным значением, т.к. различия не должны отличаться кратного в данном разрезе времени, а учесть при расчете их необходимо
# 
# 
# *	Так же в столбце количестве были нулевые или отрицательные значения. Нулевые значения пришлось убрать, по причине того, что они помешают в исследования и возникли, скорее всего по ошибки. Что касается отрицательных, то это вероятно, возвраты.
# Их из анализа убрали, т.к. цель данного исследования на данный момент являются – анализ эффективности программы лояльности
# 
# 
# *	В финальном датасете у нас получилось 64 249 записи и 10 колонок
# 

# На данном этапе подготовку данных к анализу считаем оконченной.

# ## Исследовательский анализ изученения эффективности программы лояльности

# In[21]:


#Первым делом, для себя, убедимся что, если клиент не зарегестирован в системе,
#может ли у него быть положительный статус в лояльности
data.query('customer_id == 0 and loyalty==True').shape[0]


# В данных таких записей нет.

# ### Какую прибыль принесли авторизованные клиенты и какую анонимные?
# 

# In[22]:


#считаю сумму чеков анонимных клиентов
data_amount_anon = data.query('customer_id == 0').total.sum()

#считаю сумму чека авторизированных клиентов
data_amount_auth = data.query('customer_id != 0').total.sum()
#создаю таблицу
df_amount = pd.DataFrame({'Тип пользователя': ['анонимные пользователи', 'зарегестированные пользователи'],        'Сумма покупок': [data_amount_anon, data_amount_auth]} ) 

#создадим функицю для визуализации в круговой диаграмме
def pie_client (name_index, values, text):
    
   #выдвинем меньшую часть для большей наглядности
    pull = [0]*len(values)
    pull[values.tolist().index(values.max())] = 0.1

    fig = go.Figure(data=[go.Pie(labels=name_index, values=values, pull=pull)])

    fig.update_layout(
        title=text,
        title_x = 0.5)

    fig.show() 

#использую подготовленную функцию
pie_client(df_amount['Тип пользователя'],df_amount['Сумма покупок'],           'Соотношение по прибыли между анонимными клиентами и авторизированными' )


# 11,4% пользователь длня нас остаются анонимными. Ясно, что 100% сделать невозможно, т.к. не все хотят передавать свои данные, невсегда есть карточка с собой, и невсегда есть время на регистрацию при первом посещении. Однако этот момент стоит проботать в будущем, с целью снижения этого показателя.

# ### Сколько всего клиентов и датасете?

# In[23]:


#минус один за счет 'нулевых' клиентов
print(f' В датасете присутсвует {data.customer_id.nunique()-1}  пользователей')


# ### Сколько клиентов входят в программу лояльности? 

# In[24]:


authorized_clients = data.groupby(by = 'loyalty').agg({'customer_id':'nunique'}).reset_index()

#переименовываем для легенды
authorized_clients=authorized_clients.replace(False,'Обычный клиент')
authorized_clients=authorized_clients.replace(True,'Участник программы лояльности')

#визуализируем
pie_client(authorized_clients['loyalty'],authorized_clients['customer_id'],           'Соотношение между клиентами входящими в программу лояльности и невходяшие в нее' )


# В программе участвовало 33.8%(541) клиентов входящих в программу лояльности. Грубо говоря, каждый 3-й клиент входит в привелигированую группу

# ### Как часто совершаются покупки в разных группах?

# Для начала посмотрим какой временной период представлен

# In[25]:


display(data.date.min(),data.date.max())


# Итого: старт 1 декабря 2016 года и окончание 28 февраля 2017, ровно 3 месяца. Период не самый удачный с точки зрения статистики, т.к. предновогодние продажи всегда выше средних значений. Но, работаем с тем, что имеем и посмотри на колебания покупок с течением времени

# In[26]:


#первоначальная группирова для разбивки на категории и посдчета количества покупок
count_bought = data.groupby(['date', 'loyalty']).agg({'purchase_id': 'nunique'}).reset_index()

def show_me_bar_colort(data, y, x, color,x_name, y_name, text):
    fig = px.bar(data, y=y, x=x, color=color, labels={x: x_name, y:y_name})
    title_x = 0.5
    fig.update_layout(title=text,title_x = 0.5)
    fig.show()

show_me_bar_colort(count_bought,'purchase_id', 'date','loyalty',"Дата","Количесто чеков",                  'График распределения количества чеков клиентов между группами')


# На данной визуализации виден режим работы магазинов. В обычные  дни он работает с пн по сб, вс всегда выходной. А именно перед новым годом он не работал 24 декабря по 3 января.До этих выходных магазины выходили на пиковые продажи. В целом, ежедневно, обе категории совершали покупки. 

# ### Какова разница в среднем чеке между группами?

# In[27]:


#создаем датафрем с датой, номером чека, категорией и суммой чека
check = data.groupby(['date','purchase_id','loyalty'] ).agg(total_sum=pd.NamedAgg(column='total', aggfunc="sum"),                                                        item_count=pd.NamedAgg(column='quantity', aggfunc="sum")).                                                        reset_index()

#для общего сравнения просчитаем среднее значение 2-х категорий
mean_check_category = check.groupby('loyalty').agg(mean_check=pd.NamedAgg(column='total_sum', aggfunc="mean")).reset_index().rename(columns={'loyalty': 'категория','mean_check': 'Средний чек'})

mean_check_category['Средний чек']=round(mean_check_category['Средний чек'])

#напишем функию для визулизации в виде бара
def show_me_bar(data_show, x_show, y_show, text_show):
    fig = px.bar(data_show, y=y_show, x=x_show,text=y_show)
    title_x = 0.5
    fig.update_xaxes(tickangle=45)
    fig.update_layout(
    title=text_show,title_x = 0.5)
    fig.show()
    
show_me_bar(mean_check_category, 'категория', 'Средний чек', 'График распределения среднего чека между группами')


# Как видно на графике, разница в среднем чеке ощутимая и не в пользу категории с лояльностью. Но посмотри, как она менялась с течением времени

# In[28]:


#добавляю колоник с номером недели и месяцем
check['week'] = check['date'].astype('datetime64[W]')
check['month'] = check['date'].astype('datetime64[M]')

#считаем средний чек по неделям
mean_check_week = check.groupby(['week','loyalty']).agg(mean_check_week=pd.NamedAgg(column='total_sum', aggfunc="mean")).reset_index().rename(columns={'week': 'неделя','mean_check_week': 'Величина среднего чека по недельно'})

#cоздаем функцию визуализации
def show_me_line_color(data_show, x, y, color, text_show):
    fig = px.line(data_show, x=x, y=y, color=color)
    fig.update_layout(title=text_show,title_x = 0.5)
    fig.show()

show_me_line_color(mean_check_week,'неделя','Величина среднего чека по недельно','loyalty',                   'График изменения динамики среднего чека по неделям')


# На недельном графике уже видны скачкообразные изменения категории с привелегиями. Вероятно это запуски определенных акций. Но в целом, можно обратить внимание на стандартных покупателей, у которых показатели более плавны и больше по значению.

# Доплнительно сравним средний чек по месяцам

# In[29]:


#группируем по месяцам
mean_check_month = check.groupby(['month','loyalty']).agg(mean_check_month=pd.NamedAgg(column='total_sum', aggfunc="mean")).reset_index().rename(columns={'month': 'месяц','mean_check_month': 'Величина среднего чека по месяцам'})

#округляем значение
mean_check_month['Величина среднего чека по месяцам']=round(mean_check_month['Величина среднего чека по месяцам'])

#визуализируем
def show_me_bar_colort_group(data, y, x, color, text):
    fig = px.bar(data, y=y, x=x, color=color,barmode='group', text=y)
    title_x = 0.5
    fig.update_layout(title=text,title_x = 0.5)
    fig.show()
show_me_bar_colort_group(mean_check_month,'Величина среднего чека по месяцам','месяц','loyalty',                         'График изменения среднего чека по месяцам')


# Только в феврале разница не в средних чеках не так сильно различима по сравнению с остальными данными. Но стабильно показатели хуже у привелигированной.

# ### Есть ли зависмость вежду количество товаров в чеке и вхождение в группу лояльности

# In[30]:


#использем заготовленный датафрем и посдчитаем общее среднее количество товаров в чеке
check_count = check.groupby(['loyalty'] ).agg(item_count=pd.NamedAgg(column='item_count', aggfunc="mean")).reset_index().rename(columns={'loyalty': 'категория','item_count': 'Среднее количество товаров в чеке'})

#округляем
check_count['Среднее количество товаров в чеке']=round(check_count['Среднее количество товаров в чеке'])

#Визуализируем
show_me_bar(check_count, 'категория', 'Среднее количество товаров в чеке', 'График распределения  среднего количества товаров в чеке между группами')


# Разница на лицо в с тем же контрастным результатом в пользу стандартных покупателей

# ### Сколько всего магазинов  в данных ?

# In[31]:


print(f' В датасете присутствует {data.shop_id.nunique()} магазинов')


# ### Какие магазины участвовали в данной программе?

# In[32]:


#проведем группировку по номеру магазина и количеству покупок в нем, учитывая обе группы
shop_loya = retail.query('loyalty == True').groupby(['shop_id', 'loyalty']).agg(count=pd.NamedAgg(column='purchase_id', aggfunc="count")).sort_values(by='count',ascending=False).reset_index()
shop_loya=shop_loya.shop_id.to_list()
shop_loya


# Всего в 4 магазинах использовалась программа лояльности.

# ### Какие показатели продаж в разных магазинах?

# In[33]:


#проведем группировку по номеру магазина и количеству покупок в нем, учитывая обе группы
#поним что основое количество покупок было в нулевом магазине,многократно превышающий остальные показател,
#поэтому неправильно сравнивать его с остальными 

shop_count = data.groupby(['shop_id', 'loyalty']).agg(count=pd.NamedAgg(column='purchase_id', aggfunc="nunique")).sort_values(by='count',ascending=False).reset_index()

show_me_bar_colort(shop_count.query('shop_id != "Shop 0"'),'count', 'shop_id','loyalty',"Номер магазина","Количесто чеков",                  'График распределения количества покупок клиентов между группами во всех магазинах')


# Посмотри поближе магазины с привелегиями

# In[34]:


show_me_bar_colort(shop_count.query('shop_id in @shop_loya and shop_id != "Shop 0"'),'count', 'shop_id','loyalty',"Номер магазина","Количество чеков",'График распределения количества покупок клиентов между группами в магазинах с привелегиями')


# В 8 магазине очень малый процент чеков из группы лояльсноти, в 19 ровно пополам, в 28 все чеки из привелигированной группы, скорее всего к последнем доступ стандартным клиентам закрыт
# 
# Посмотрим как распределены клиенты в нулевом

# In[35]:


show_me_bar_colort(shop_count.query('shop_id == "Shop 0"'),'count', 'shop_id','loyalty',"Номер магазина","Количество чеков",                  'График распределения количества чеков клиентов между группами нулевом магазине в шт ')

pie_client(shop_count.query('shop_id == "Shop 0"')['loyalty'],shop_count.query('shop_id == "Shop 0"')['count'],           'График распределения количества чеков клиентов между группами нулевом магазине в %' )


# Т.к. это основной магазин, он и дал похожее распределение как и в  шаге 2.3

# ### Самые преданные покупатели, сколько раз они совершили покупки?

# In[36]:


#Создаем датасет с рейтингом клиентов зависящий от количества чеков
clients =data.query('customer_id != 0').groupby(['customer_id','loyalty']).                                                        agg(count_check=pd.NamedAgg(column='purchase_id', aggfunc="nunique")).                                                        sort_values(by='count_check',ascending=False).reset_index().                                                        rename(columns={'count_check': 'Количество чеков'})

#посмотри на распределение
fig = px.box(clients, y="Количество чеков")
fig.update_layout(title='Диаграма рассеяния количества чеков между всеми покупателями',title_x = 0.5)
fig.show()


# In[37]:


clients.head()


# Клиент с номером 23529 оказался самым частым клиентом, да и еще состоит в группе лояльности. 

# ### Какие товары  самые популярные?

# In[38]:


prodact_rait = data.groupby(['item_id','price_per_one']).agg(count=pd.NamedAgg(column='item_id', aggfunc="count"),    suma=pd.NamedAgg(column='price_per_one', aggfunc="sum")).sort_values(by='count',ascending=False).reset_index().head()

prodact_rait_list = prodact_rait.item_id.to_list()
prodact_rait


# Как часто эти товары покапуют обе группы?

# In[52]:


prodact_top = data.query('item_id in @prodact_rait_list').groupby('loyalty').agg(count=pd.NamedAgg(column='loyalty', aggfunc="count")).reset_index().rename(columns={'loyalty': 'категория','count': 'Количество'})

show_me_bar(prodact_top, 'категория', 'Количество', 'График распределния продаж топ-товаров по категориям клиентов')


# В целом на это и расчитана стратегия проведения акции - как раз таки продвинуть плохо продаваемы товары, а не те, которые даже у стандартных клиентов лучше всего идут. 

# ### Расчет значения LTV
# Мы можем подсчитать обороты, какую валовую прибыл приносит 1 клиент разных категорий.
# 

# In[40]:


#считаем сколько каждая группа принесал прибыли в общем и сколько уникальных клиентов было в зависмости от категории
gross_profit = data.groupby(['loyalty']).agg(profit=pd.NamedAgg(column='total', aggfunc="sum"),                                             count_client=pd.NamedAgg(column='customer_id', aggfunc="nunique")).reset_index().                                        rename(columns={'loyalty': 'категория','ltv': 'Показатель ltv'})

#Делим валовбу прибыл целой катетогории на количество клиентов получая значение LTV
gross_profit['Показатель ltv'] =round(gross_profit['profit']/gross_profit['count_client'])

#визуализируем
show_me_bar(gross_profit, 'категория', 'Показатель ltv', 'График LTV')


# И тут, в очередной раз, группа лояльности остается в аутсайдерах.

# In[41]:


#Вычисляю первый день, когда клиент засветился
session_start = data.groupby(['customer_id']).agg({'date':'min'}).reset_index().rename(columns={'date':'session_start'})
#подтягиваю дату к основной таблице
session = data.merge(session_start, how='left', on='customer_id')
#переименовую дату на более внятную
session = session.rename(columns = {'date':'event_dt'})

#создаю профиль пользователей, с номером id, категорий и первой датой
profiles = data.sort_values(by=['customer_id', 'date']).groupby('customer_id').agg({'date': 'first','loyalty': 'first'})        .rename(columns={'date': 'first_ts'}).reset_index()

#меняю форматы
profiles['date'] = profiles['first_ts'].dt.date
profiles['month'] = profiles['first_ts'].astype('datetime64[M]')

#устанавливаю горизон анализа
last_suitable_acquisition_date = datetime(2017, 2, 28).date()

#количество дней
horizon_days=60

# исключаем пользователей, не «доживших» до горизонта анализа
if not False:
    last_suitable_acquisition_date = last_suitable_acquisition_date - timedelta(horizon_days)
result_raw = profiles.query('date <= @last_suitable_acquisition_date')

#Добавить данные о покупках в профили
result_raw = result_raw.merge(session[['customer_id', 'event_dt', 'total']],on='customer_id',how='left')
result_raw['lifetime'] = (result_raw['event_dt'] - result_raw['first_ts']).dt.days


#Строим таблицу выручки строим «треугольную» таблицу
result = result_raw.pivot_table(index='loyalty',columns='lifetime',values='total',  aggfunc='sum')

#Считаем сумму выручки с накоплением
result = result.fillna(0).cumsum(axis=1)

#Объединяем размеры когорт и таблицу выручки
cohort_sizes = result_raw.groupby('loyalty').agg({'customer_id': 'nunique'}).rename(columns={'customer_id': 'cohort_size'})

#Считаем LTV делим каждую «ячейку» в строке на размер когорты
result = cohort_sizes.merge(result, on=['loyalty'], how='left').fillna(0)
#исключаем все лайфтаймы, превышающие горизонт анализа
result = result.div(result['cohort_size'], axis=0).reset_index()

dimensions='loyalty'

#исключаем все лайфтаймы, превышающие горизонт анализа
result = result[['cohort_size'] + list(range(horizon_days))]

#для таблицы динамики LTV убираем 'cohort' из dimensions
if 'cohort' in dimensions:
        dimensions = []   
        
result.reset_index()


# In[42]:


report = result.drop(columns=['cohort_size'])
report.T.plot(grid=True, figsize=(20, 10), xticks=list(report.columns.values))
plt.title('LTV с разбивкой по дням')
plt.ylabel('Уровень LTV')
plt.xlabel('Лайфтайм')
plt.legend(['Стандартная группа', 'Привелигированная'],fontsize=20, shadow=True)

plt.show()


# Уровень LTV у стандартной категории многократно превышает уровень группы лояльности. Если брать в расчет, что привелигированные клиента хоть и платят за подписку 200 руб, но у нас нет данных о величине скидки котору магазины предоставляют им.

# ## Проверка статистических гипотез
# Анализирую аномалии,провожу группировку и расчитываю средние показатели для каждого id

# ### Посчитаем 95-й и 99-й перцентили стоимости товара

# In[43]:


perc_price = np.nanpercentile(data.price_per_one, [95,99])
perc_price[0]


# Для А-Б теста возьем за верхнее значаение в цене 8.35

# ### Посчитаем 95-й и 99-й перцентили количества товаров в чеке

# In[44]:


perc_count = np.nanpercentile(data.quantity, [95,99])
perc_count[0]


# Для А-Б теста возьем за верхнее значаение по количетсву товаров в чеке 11

# ### Готовим группы

# In[45]:


#Фильтруем данные
data=data.query('price_per_one < @perc_price[0] and quantity < @perc_count[0]')

#делаем общий датафрем с номером чека, категории лояльсноти, суммой покупки и количество товара
data_group = data.groupby(['purchase_id','loyalty']).agg({'total':'sum','quantity':'sum' }).reset_index()

#выделяем 2 группа. а будет в категории лояльности
group_a = data_group.query('loyalty == True')
group_b = data_group.query('loyalty == False')

display(group_a,group_b)


# ### Как распределены выборки по сумме чека?
# Проверка на нормальность через histplot

# In[46]:


fig, ax = plt.subplots(1 ,2, figsize=(15,5))
sns.histplot(group_a.total, ax=ax[0],kde=True,alpha=0.3,bins=10)
sns.histplot(group_b.total, ax=ax[1],kde=True,alpha=0.1,bins=10)
ax[0].set(ylabel = 'Сумма в чеке', xlabel = 'Количество значений', title = 'Группа а' )

ax[1].set(ylabel = 'Сумма в чеке', xlabel = 'Количество значений', title = 'Группа b')
plt.xlim(-1000, 10000);
plt.show()


# И проверка через qq-графики

# In[47]:


plt.figure(figsize=(12,8))
plt.subplot(2,2,1)
st.probplot(group_a.total, dist = 'norm', plot=plt)
plt.subplot(2,2,2)
st.probplot(group_b.total, dist = 'norm', plot=plt);


# ### Как распределены выборки по количеству товаров в чеке?
# Проверка на нормальность через histplot

# In[48]:


fig, ax = plt.subplots(1 ,2, figsize=(15,5))
sns.histplot(group_a.quantity, ax=ax[0],kde=True,alpha=0.3,bins=10)
sns.histplot(group_b.quantity, ax=ax[1],kde=True,alpha=0.1,bins=10)
ax[0].set(ylabel = 'Количество товаров в чеке', xlabel = 'Количество значений', title = 'Группа а' )

ax[1].set(ylabel = 'Количество товаров в чеке', xlabel = 'Количество значений', title = 'Группа b')
plt.xlim(-1000, 10000);
plt.show()


# И проверка через qq-графики

# In[49]:


plt.figure(figsize=(12,8))
plt.subplot(2,2,1)
st.probplot(group_a.quantity, dist = 'norm', plot=plt)
plt.subplot(2,2,2)
st.probplot(group_b.quantity, dist = 'norm', plot=plt);


# Вывод: отклонения слишком велики чтобы считать их нормальными, поэтому для проверки гипотез будем применять критерий Манна-Уитни со стандартным критерим статистической значимости 0.05

# ### Средний чек участников программы лояльности выше, чем у остальных покупателей.
# 
# * За нулевую гипотезу Н0  - 'Разницы в среднем чеке между группами нет, суммы равны'
# 
# 
# * За алтернативную гипотезу Н1 - 'Разница в среднем чеке мужду группами есть, суммы различны'

# In[50]:


# критический уровень статистической значимости
alpha = .05

results = st.mannwhitneyu(group_a.total, group_b.total,True) # ваш код

print('p-значение: ', results.pvalue)

if results.pvalue < alpha:
     print('Отвергаем нулевую гипотезу: разница статистически значима')
else:
    print(
        'Не получилось отвергнуть нулевую гипотезу, вывод о различии сделать нельзя'
    ) 


# Тест дает результат: разница статистически значима, т.е. нулевую гипотезу о равенстве чеков между группами мы отвергаем, они различны. Что и подтверждается в вычислениях в предъидущих пунктах

# ### Среднее количесто товаров в чеке у участников программы лояльности выше, чем у остальных покупателей
# 
# * За нулевую гипотезу Н0  - 'Разницы в среднем количестве товаров в чеке нет.'
# 
# 
# * За алтернативную гипотезу Н1 - 'Разница в среднем количестве товаров в чеке есть'

# In[51]:


results = st.mannwhitneyu(group_a.quantity, group_b.quantity,True) # ваш код

print('p-значение: ', results.pvalue)

if results.pvalue < alpha:
     print('Отвергаем нулевую гипотезу: разница статистически значима')
else:
    print(
        'Не получилось отвергнуть нулевую гипотезу, вывод о различии сделать нельзя'
    ) 


# Принимаем альтернативную гипотезу: среднее количество товаров чеке в группе а и б разное. 

# ## Итоги

# **Общие выводы**
# 
# * В датасете присутствует 1564  уникальных покупателя из которых всего 541 (33.8%) являются группой лояльности. 
# 
# 
# *	О 30 магазинах была данные в датасете. Всего 4 из них были покупатели входящие в программу лояльности. 1 магазин самый крупный, в нем сосредоточена бОльшая часть записей. Скорее всего это или интернет-магазин, или основной магазин в городе миллионнике. Так же был магазин, в которым вообще не было стандартных покупателей, возможно вход в него исключительно по карточкам привилегированных покупателей.
# 
# 
# *	Не авторизированные пользователи принесли 11,4% от всей общей прибыли
# 
# *	Временно интервал в данных с 1 декабря 2016 года по 28 февраля 2017. Что является непоказательным, т.к. в предновогодние праздники всегда возникает ажиотаж, связанный с поисками подарков, желание закончить дела в уходящем году и тд. К тому же выяснилось, что режим работы магазинов с понедельника по субботу, а данном интервале есть большой перерыв с 24 декабря по 3 января.
# 
# 
# *	Средний чек сильно отличает в пользу стандартны клиентов
# 
# 
# *	Среднее количество товаров в чеке так же больше у стандартных клиентов
# 
# 
# *	В среднем 1 клиент из стандартно группу приносит порядком больше валовой прибыли в отличие от привилегированной группы
# 

# **Заключение**
# 
# В целом данные для исследования выбраны неудовлетворительно, т.к. временной период выбран максимально не удачно, а так же система лояльности введена не во всех магазинах. Но всем данным, которые у нас имеются можно сделать вывод, что систем лояльности не приносит желаемых результатов ни по одной из исследуемых метрик. Рекомендуется или выбрать другой промежуточный период для исследование, или прекратить данную программу и сосредоточить  усиления на другом способе продвижения.

import pandas as pd
import numpy as np
import math
import random

Sub_mas = []

print('Extracting data')

with open('D:/Kek_Kok_Kik/Drones delivery_compet/busy_day.in') as file:
    line_list = file.read().splitlines()

ROWS, COLS, DRONES_num, TURNS, MAXLOAD = map(int, line_list[0].split())
SCORE = 0
# веса продуктов каждого типа (делает таблицу)
weights = line_list[2].split()
products_df = pd.DataFrame({'weight': weights})
# получение числа из таблицы
# a = int(products_df.iloc[3])
# print(a)

# количество складов
wh_count = int(line_list[3])
wh_endline = (wh_count * 2) + 4

# количество товаров каждого типа на складе
wh_invs = line_list[5:wh_endline + 1:2]
for i, wh_inv in enumerate(wh_invs):
    products_df[f'wh{i}_inv'] = wh_inv.split()

# products_df has shape [400,11]
# (# of products, [weight, wh0_inv, wh1_inv,...])
products_df = products_df.astype(int)

# расположение складов
wh_locs = line_list[4:wh_endline:2]
wh_rows = [wl.split()[0] for wl in wh_locs]
wh_cols = [wl.split()[1] for wl in wh_locs]

warehouse_df = pd.DataFrame(
    {'wh_row': wh_rows, 'wh_col': wh_cols}).astype(np.uint16)

order_locs = line_list[wh_endline + 1::3]
o_rows = [ol.split()[0] for ol in order_locs]
o_cols = [ol.split()[1] for ol in order_locs]

orders_df = pd.DataFrame({'row': o_rows, 'col': o_cols})

orders_df[orders_df.duplicated(keep=False)].sort_values('row')

orders_df['product_count'] = line_list[wh_endline + 2::3]

order_array = np.zeros((len(orders_df), len(products_df)), dtype=np.uint16)
orders = line_list[wh_endline + 3::3]

for i, order in enumerate(orders):
    products = [int(prod) for prod in order.split()]
    for p in products:
        order_array[i, p] += 1

df = pd.DataFrame(data=order_array,
                  columns=['p_' + str(i) for i in range(2000)],
                  index=orders_df.index)

# таблица с номерами заказов
orders_df = orders_df.astype(int).join(df)
print('... success')


def distance(point1, point2):
    return np.sqrt((point1[0] - int(point2[0])) ** 2 + (point1[1] - int(point2[1])) ** 2)


class Order(object):
    def __init__(self, row, colom, number):
        self.number = number
        self.point = [row, colom]
        self.products = []
        self.haveabind = False
        self.warehouse = []
        pass
    def setProduct(self, products):
        a = [int(x) for x in products.split()]
        self.products = a
        pass
    # присваивает заказу ближайший склад
    def setWarehouse(self, locs_wh):
        dist = [distance(self.point, [wl.split()[0], wl.split()[1]]) for wl in locs_wh]
        self.warehouse = np.argmin(dist)
        pass
    def setWarehouse_radial(self, wh_massive):
        self.warehouse = []
        for wh in range(len(wh_massive)):
            rast = distance(self.point, [wh_massive[wh].point[0], wh_massive[wh].point[1]])
            if rast - wh_massive[wh].radius <= 0:
                self.haveabind = True
                self.warehouse.append(wh)
        #if len(self.warehouse) > 1:
            #print(self.warehouse, "kkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk")
        pass


class Warehouse(object):
    def __init__(self, row, colom):
        self.point = [row, colom]
        self.orders = []
        self.products = []
        self.W = [[]]
        self.radius = 700
        pass
    def setOrder(self, order):
        can = False
        for i in order.products:
            for j in range(len(self.products)):
                if i == j and self.products[j] > 0:
                    can = True
        if can:
            self.orders.append(order)
        pass
    def setProducts(self, products_num):
        self.products = [int(k) for k in products_num]
        pass
    # разделяет продукты на те что перевезти на другие склады те что останутся для доставки
    def determProducts(self):
        self.requirement = [0 for _ in range(len(self.products))]
        for i in self.orders:
            for j in i.products:
                self.requirement[j] += 1
        self.div = [int(self.products[k]) - self.requirement[k] for k in range(len(self.products))]
        self.not_requirement = [0 for _ in range(len(self.products))]
        for i in range(len(self.div)):
            if (self.div[i] >= 0):
                self.requirement[i] = 0
                self.not_requirement[i] = self.div[i]
                self.products[i] -= self.not_requirement[i]
            else:
                self.requirement[i] = abs(self.div[i])
                #self.products[i] = 0
        pass
    def determProducts0(self):
        self.requirement = [0 for _ in range(len(self.products))]
        self.not_requirement = [0 for _ in range(len(self.products))]
        to_delive = [0 for _ in range(len(self.products))]
        temp_all_prod_ord = [0 for _ in range(len(self.products))]
        for i in self.orders:
            for j in i.products:
                temp_all_prod_ord[j] += 1
        div = [self.products[a] - temp_all_prod_ord[a] for a in range(len(self.products))]
        
        for a in range(len(div)):
            if div[a] < 0:
               self.requirement[a] = abs(div[a])
            else:
                self.not_requirement[a] = div[a]
                to_delive[a] = temp_all_prod_ord[a]

        self.products = to_delive
        pass
    def show_state(self, it = -1):
        sum_p = 0
        for i in self.products:
            sum_p += i
        sum_r = 0
        for i in self.requirement:
            sum_r += i
        sum_nr = 0
        for i in self.not_requirement:
            sum_nr += i
        print("\t", it, ") есть ", sum_p, "; требуется ", sum_r, "; не требуется ", sum_nr)
    def products_num(self):
        sum_p = 0
        for i in self.products:
            sum_p += i
        return sum_p
    def requirement_num(self):
        sum_r = 0
        for i in self.requirement:
            sum_r += i
        return sum_r
    def createW(self):
        self.W = [[0 for _ in range(len(self.orders))] for _ in range(len(self.orders))]
        for i in range(len(self.orders)):
            for j in range(len(self.orders)):
                if i == j:
                    self.W[i][j] = -1
                    continue
                self.W[i][j] = math.ceil(distance(self.orders[i].point, self.orders[j].point))
        pass

doneOrdersyyy = 0
class Drone(object):
    def __init__(self, point, isdeliver, number, wh=None):
        self.number = number
        self.point = point
        self.turns = TURNS
        self.isdeliver = isdeliver
        self.whatorderdoIdelive = -1  # мб не нужно
        if isdeliver == True:
            self.wh = wh
        pass
        self.bobo = 0
        self.momo = 0
    def delive_cicle(self):
        self.rar = True
        self.wh_mass[self.wh_numb].determProducts()
        if self.wh_mass[self.wh_numb].products_num() == 0:
            #if self.wh_mass[self.wh_numb].requirement_num() == 0:
                maxi = 0
                for i in range(len(self.wh_mass)):
                    if self.wh_mass[i].products_num() > self.wh_mass[maxi].products_num():
                        maxi = i
                self.turns -= math.ceil(distance(self.wh_mass[self.wh_numb].point, self.wh_mass[maxi].point))
                #print("Дрон ", self.number, " сменил склад ", self.wh_numb, " на склад ", maxi)
                self.wh_numb = maxi
                if self.wh_mass[self.wh_numb].products_num() == 0 and self.wh_mass[self.wh_numb].requirement_num() == 0:
                    return
            # else:
            #     #self.wait(1)
            #     if self.rar:
            #         print(" Ход ", TURNS - turn_simul, " номер ", self.number, " Закончил в городе ", self.wh_numb)
            #         self.rar = False 
            #     return
        if self.wh_mass[self.wh_numb].products_num() == 0:
            return
        #self.wh_mass[self.wh_numb].show_state()
        way = self.genetic_solve(2)
        #if way == [116, 93, 284, 5, 167]:
        #print(way)
        if len(way) <= 4:
            self.momo += 1
        else: self.bobo += 1
        self.turns -= self.f_way(way)
        bag = 0
        temp_sub_mas = []
        products_temp_global = []
        for i in way:
            steps = 0
            products_temp = []
            for o_i in range(len(self.wh_mass[self.wh_numb].orders[i].products)):
                if MAXLOAD <= bag:
                    if MAXLOAD < bag: print("ошибка, перевес")
                    break
                o = self.wh_mass[self.wh_numb].orders[i].products[o_i]
                if self.wh_mass[self.wh_numb].products[o] != 0 and MAXLOAD >= bag + weights_int[o]:
                    da = True
                    for om in products_temp:
                        if (om == o):
                            da = False
                    net = True
                    for om in products_temp_global:
                        if (om == o):
                            net = False
                    products_temp.append(o)
                    products_temp_global.append(o)
                    self.wh_mass[self.wh_numb].products[o] -= 1  # забрали
                    self.wh_mass[self.wh_numb].orders[i].products[o_i] = -1  # мгновенно доставили :)
                    if da:
                        if net:
                            self.turns -= 2  # будем по 1 продукту доставлять и брать
                        else:
                            self.turns -= 1
                    bag += weights_int[o]
                    Sub_mas.append(str(self.number) + " " + 'L' + " " + str(self.wh_numb) + " " + str(o) + " " + str(1))
                    temp_sub_mas.append(str(self.number) + " " + 'D' + " " + str(
                        self.wh_mass[self.wh_numb].orders[i].number) + " " + str(o) + " " + str(1))
            for o_i in range(len(self.wh_mass[self.wh_numb].orders[i].products) - 1, -1, -1):
                if self.wh_mass[self.wh_numb].orders[i].products[o_i] == -1:
                    #print(self.wh_mass[self.wh_numb].orders[i])
                    self.wh_mass[self.wh_numb].orders[i].products.pop(o_i)
                    #print(len(self.wh_mass[0].orders))
                    if len(self.wh_mass[self.wh_numb].orders[i].products) == 0:
                        global SCORE, doneOrdersyyy
                        doneOrdersyyy += 1
                        SCORE += math.ceil((self.turns) / TURNS * 100)
        Sub_mas.extend(temp_sub_mas)
        if Sub_mas[-1] == ['']: print("пустая сторока")
        pass
    def deliver__init__(self, wh_mass, start_wh):
        self.wh_mass = wh_mass
        self.wh_numb = start_wh
        pass
    def delive(self, order_num, products):
        order_coords = orders_mass[order_num].point
        self.__goto__(order_coords)

        for prod_num in products:
            if self.__nessesary_delive_this_prod__(prod_num, order_num):
                # self.orders_mass[prod_num] -= 1
                self.bag.append(prod_num)
            else:
                print("Доставялешь то, что не надо доставлять!")
        pass
    def load(self, wh_num, products_to_load):
        wh_coords = wh_mass[wh_num].point
        self.__goto__(wh_coords)

        for prod_num in products_to_load:
            count = wh_mass[wh_num].not_requirement[prod_num]
            if count <= 0:
                print("Берешь неберущееся!")
            else:
                if self.__can_add_to_bag__(prod_num):
                    wh_mass[wh_num].not_requirement[prod_num] -= 1
                    self.bag.append(prod_num)
        pass  
    def wait(self, wait_turns):
        self.turns -= wait_turns
        Sub_mas.append(str(self.number) + " " + 'W' + " " + str(wait_turns))
        pass  
    def __goto__(self, point):
        dist = math.ceil(distance(self.point, point))
        self.turns -= dist
        # проверочка буить на оставшееся количество ходов
        self.point = point
        pass
    def __can_add_to_bag__(self, oneprod):
        if int(weights[oneprod]) + self.nowload <= MAXLOAD:
            return 1
        else:
            return 0
    def __can_pop_from_bag__(self, oneprod):
        for i in self.bag:
            if i == oneprod:
                return i
        return -1
    def __nessesary_delive_this_prod__(self, oneprod, order_num):
        for i in orders_mass[order_num]:
            if i == oneprod:
                return 1
        return 0
    
    def checkfororders(self):
        pass

    # часть с генетикой
    def f_way(self, x):
        sum = 0
        m = len(x)
        # print(m)
        sum += math.ceil(distance(self.wh_mass[self.wh_numb].point, self.wh_mass[self.wh_numb].orders[x[0]].point))
        for i in range(m - 1):
            sum += self.wh_mass[self.wh_numb].W[x[i]][x[i + 1]]
        sum += math.ceil(distance(self.wh_mass[self.wh_numb].point, self.wh_mass[self.wh_numb].orders[x[m - 1]].point))
        # раскоменнтить ради эксперемента
        # return sum / m
        return sum
    def f(self, x):
        count = 0
        bag = 0
        wh_products = self.wh_mass[self.wh_numb].products.copy()
        for i in x:
            steps = 0
            products_temp = self.wh_mass[self.wh_numb].orders[i].products.copy()
            for o_i in range(len(products_temp)):
                if MAXLOAD <= bag:
                    if MAXLOAD < bag: print("ошибка, перевес")
                    break
                o = products_temp[o_i]
                if wh_products[o] != 0 and MAXLOAD >= bag + weights_int[o]:
                    wh_products[o] -= 1  # забрали
                    products_temp[o_i] = -1  # мгновенно доставили :)
                    bag += weights_int[o]
            for o_i in range(len(products_temp) - 1, -1, -1):
                if products_temp[o_i] == -1:
                    products_temp.pop(o_i)
                    if len(products_temp) == 0:
                        count += 1

        sum = 0
        m = len(x)
        # print(m)
        sum += math.ceil(distance(self.wh_mass[self.wh_numb].point, self.wh_mass[self.wh_numb].orders[x[0]].point))
        for i in range(m - 1):
            sum += self.wh_mass[self.wh_numb].W[x[i]][x[i + 1]]
        sum += math.ceil(distance(self.wh_mass[self.wh_numb].point, self.wh_mass[self.wh_numb].orders[x[m - 1]].point))
        # раскоменнтить ради эксперемента
        # return sum / m
        if count != 0:
            return sum / (count+1)
        return sum
    def mutation(self, x):
        m = len(x)
        k = random.randint(0, m - 2)
        x[k], x[k + 1] = x[k + 1], x[k]
    def crossing(self, pop, prob=20):
        n = int(len(pop) / 2)
        for i in range(n):
            pop[n + i] = pop[i].copy()
            m = len(pop[i])
            if (m <= 1):
                continue
            r = random.randint(0, m - 1)
            l = random.randint(0, m - 1)
            # if m != 1:
            while r == l:
                l = random.randint(0, m - 1)
            if l < r:
                l, r = r, l
            for j in range(math.ceil((l - r) / 2)):
                pop[i + n][r + j], pop[i + n][l - j] = pop[i + n][l - j], pop[i + n][r + j]
            if (1 + random.randint(0, 99) <= prob):
                self.mutation(pop[i + n])
    def qsort(self, pop):
        listik = []
        for i in range(len(pop)):
            listik.append((self.f(pop[i]), i))
        listik.sort(key=lambda x: x[0])
        newpop = []
        for i in range(len(pop)):
            newpop.append(pop[listik[i][1]].copy())
        for i in range(len(pop)):
            pop[i] = newpop[i].copy()
        pass
    def randPopulation_delive(self, pop):
        n = len(pop)
        m = len(self.wh_mass[self.wh_numb].orders)

        for i in range(n):
            temp_prod = self.wh_mass[self.wh_numb].products.copy()
            bag = 0
            pop[i] = []
            for _ in range(m):
                opop = m
                #print(len(self.wh_mass[self.wh_numb].orders))
                while (True):
                    opop -= 1
                    new_o = random.randint(0, len(self.wh_mass[self.wh_numb].orders) - 1)
                    #print(new_o)
                    myb = True
                    for p in pop[i]:
                        if new_o == p:
                            myb = False
                    if myb:
                        break
                    #print(pop[i])
                    #print("len order ", len(self.wh_mass[self.wh_numb].orders))
                    #print("заказы")
                    #for o in self.wh_mass[self.wh_numb].orders:
                        #print(o.products)
                    deb = []
                    for h in range(len(temp_prod)):
                        if temp_prod[h] == 0:
                            deb.append(0)
                        else: deb.append(h)
                    #print("продукты ", deb)
                #print("я вышел ")                
                temp_attantion = False
                for o in self.wh_mass[self.wh_numb].orders[new_o].products:
                    if MAXLOAD <= bag:
                        break
                    if temp_prod[o] != 0 and MAXLOAD >= bag + weights_int[o]:
                        temp_prod[o] -= 1
                        bag += weights_int[o]
                        temp_attantion = True
                        # раскомментить/закомментить break, если мы хотим брать только по одному/много продукту
                        # break
                if temp_attantion:
                    pop[i].append(new_o)
            # опасная штука. нулевые получатся если не успел нарандомить, когда мало. поэтому вставляем насильно(не идеал)
            if len(pop[i]) == 0:
                for ord in range(len(self.wh_mass[self.wh_numb].orders)):
                    for pr in self.wh_mass[self.wh_numb].orders[ord].products:
                        if self.wh_mass[self.wh_numb].products[pr] != 0:
                            pop[i].append(ord)
                            break
        for i in range(len(pop) - 1, -1, -1):
            if len(pop[i]) == 0:
                pop.pop(i)
        if len(pop) == 0:
            sum = 0
            for it in self.wh_mass[self.wh_numb].products:
                sum += it
            print("осталось ", sum, " ", self.wh_numb)
        pass 
    def genetic_solve(self, max_osob):
        N = max_osob  # число особей в популяции
        M = 50  # максимальное число городов
        prob = 20  # вероятность мутации
        if N % 2 != 0:
            print("think again")
        pop = [[0 for _ in range(M)] for _ in range(N)]
        T = 10  # число поколений
        self.randPopulation_delive(pop)

        for t in range(T):
            self.crossing(pop, prob)
            self.qsort(pop)

        return pop[0]
    


# создаем Заказы предавая им координаты
orders_mass = [Order(int(order_locs[ol].split()[0]), int(order_locs[ol].split()[1]), ol) for ol in range(len(order_locs))]

# вставляем в Заказы продукты
for i in range(len(orders_mass)):
    orders_mass[i].setProduct(orders[i])
# создаем Склады предавая им координаты
wh_mass = [Warehouse(int(wl.split()[0]), int(wl.split()[1])) for wl in wh_locs]
# вставляем в Склады продукты
for i in range(len(wh_mass)):
    wh_mass[i].setProducts(wh_invs[i].split())





# for i in orders_mass:
#     i.setWarehouse(wh_locs)
#     wh_mass[i.warehouse].setOrder(i)
for i in orders_mass:
    i.setWarehouse_radial(wh_mass)
    if len(i.warehouse) != 0:
        for tete in i.warehouse:
            wh_mass[tete].setOrder(i)
#print(len(wh_mass[0].orders))
# на каждом складе разбиваем продукыты на 3 категории и составляем матрицу весов
for i in wh_mass:
    i.determProducts()
    i.createW()



weights_int = [int(w) for w in weights]


raspeadDR = [4,1,2,4,5,2,5,2,1,4]
if len(raspeadDR) != len(wh_mass): print("Ошибка, блин")
summir = 0
for i in raspeadDR:
    summir += i
if summir != DRONES_num: print("Ошибка, блин ", summir - DRONES_num)

# создаем дронов 2х типов
DRONES_delivers_count = DRONES_num
# дроны разлетаются по своим начальным Складам
beta = 0
DRONES_delivers = [Drone(wh_mass[0].point, True, nb) for nb in range(DRONES_delivers_count)]
for i in range(DRONES_delivers_count):
    DRONES_delivers[i].__goto__(wh_mass[beta].point)
    DRONES_delivers[i].deliver__init__(wh_mass, beta)
    beta += 1
    if beta >= len(wh_mass): beta = 0

# основной цикл стимуляции
print("Симуляция начата")

for turn_simul in range(TURNS,0,-1):
    if turn_simul % 1000 == 0 or turn_simul == TURNS:
        print(turn_simul/1000, " ", SCORE)
        #for i in range(len(wh_mass)):
            #wh_mass[i].show_state(i)
    # if SCORE > 90000:
    #print("итерация ", turn_simul, " Счет: ", SCORE)
    # sum1 = 0
    # sum2 = 0
    #for i in range(len(wh_mass)):
        #wh_mass[i].show_state(i)
        #sum1 += wh_mass[i].requirement_num()

    # цикл по доставщикам
    for i in DRONES_delivers:
        # проверяется ход дрона
        if i.turns == turn_simul:
            # производится отправление на очередную петлю (вызов deliver_cicle)
            i.delive_cicle()

    continue

print("Симуляция окончена")
print("SCORE ", SCORE)
for i in wh_mass:
    print("Осталось ", i.requirement_num())
for i in DRONES_delivers:
    print("меньше или равно 2: ", i.momo, " больше 2: ", i.bobo)
print("Заказов выполнено: ", doneOrdersyyy)

print("Запись...")
submis = open('D:/Kek_Kok_Kik/Drones delivery_compet/submission.csv', 'w')
submis.write(str(len(Sub_mas)) + '\n')
for i in Sub_mas:
    submis.write(i + '\n')
print("окончена:)")
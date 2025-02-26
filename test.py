import matplotlib.pyplot as plt
import numpy as np

data = {
    "low_order": [
        [1733135894447, 95053.5, 1733143127162, 95339.4, 0.0019987602187552253],
        [1733143966869, 94986.9, 1733160686063, 95272.6, 0.0019987635479666874],
        [1733437288639, 98232.6, 1733437601175, 97937.8, -0.004010073740680408],
        [1733437606450, 97533.9, 1733437610066, 97241.2, -0.0040100410114231036],
        [1733437684593, 95937.7, 1733437687544, 95649.5, -0.004013084229400054],
        [1733437689828, 95486.2, 1733437692570, 95199.7, -0.00400946326511542],
        [1733437692571, 95285.2, 1733437694056, 94999.3, -0.004009495859443101],
        [1733437694214, 95081.6, 1733437694895, 94796.3, -0.004009611134611845],
        [1733437694993, 94747.9, 1733437695650, 94462.6, -0.004020242932123329],
        [1733437699727, 93616.7, 1733437700076, 93333.6, -0.004033205619412449],
        [1733437701752, 93155.9, 1733437704147, 92876.4, -0.004009375901736],
        [1733437706800, 92714.9, 1733437707558, 92436.4, -0.004012882371014047],
        [1733437712372, 91817.8, 1733437713033, 91542.0, -0.004012824714338835],
        [1733437713036, 91932.0, 1733437713036, 91478.7, -0.005955251878306033],
        [1733437713036, 91934.0, 1733437713036, 91475.6, -0.006011172378207807],
        [1733437713036, 91934.0, 1733437713036, 91474.3, -0.006025455237154031],
        [1733437713036, 91934.0, 1733437713036, 91472.9, -0.006040837231573558],
        [1733437719869, 91070.8, 1733437719870, 90366.8, -0.008790471721915494],
        [1733437719873, 91070.8, 1733437719875, 90694.0, -0.005154629854235203],
        [1733437720015, 91076.6, 1733437720225, 90800.1, -0.004045150831331611],
        [1733758962931, 97754.3, 1733760763354, 98048.0, 0.0019954716057440933],
        [1733773411961, 97231.0, 1733775561908, 96939.1, -0.00401116886787689],
        [1733848707074, 94926.0, 1733848898830, 94641.1, -0.0040103200406588204],
        [1733852259217, 94660.2, 1733852747737, 94944.3, 0.0019922807372323445],
        [1734552495039, 101395.4, 1734552500891, 101091.2, -0.004009164002405695],
        [1734552521038, 100996.9, 1734552538089, 100693.8, -0.004010115816465264],
        [1734563156409, 100480.8, 1734563523129, 100783.0, 0.0019985215760593986],
        [1734565767360, 100238.6, 1734566017529, 100539.4, 0.0019918618969279054],
        [1734566140596, 100149.4, 1734568922439, 100450.7, 0.0019994813376114093],
        [1734573245297, 100094.7, 1734573485205, 99794.4, -0.004009186888242255],
        [1734573490590, 99819.7, 1734574285755, 99520.2, -0.004009439289711979],
        [1734574314936, 99315.1, 1734574368874, 99017.1, -0.0040095811733528155],
        [1734629438818, 99357.8, 1734629789882, 99059.7, -0.004009296414182572],
        [1734629789902, 99110.3, 1734630201344, 98812.9, -0.004009728486867758],
        [1734630568578, 98588.2, 1734630612555, 98884.4, 0.001995416870608447],
        [1734630717089, 98414.3, 1734630774181, 98119.0, -0.004009610778748284],
        [1734630775344, 98129.5, 1734630798737, 98424.4, 0.0019962082573019577],
        [1734630877966, 97953.5, 1734630880863, 97659.2, -0.004013540966954498],
        [1734637535141, 96692.6, 1734637646263, 96402.5, -0.004009258058660258],
        [1734639075987, 95778.5, 1734640547438, 96066.6, 0.0019989611373776315],
        [1734693634628, 94075.1, 1734693713451, 93792.5, -0.004013034091212008],
        [1734694528788, 93114.3, 1734694583717, 92834.9, -0.004009644002417211],
        [1734967512418, 93356.5, 1734968331183, 93636.7, 0.0019924164350089155],
        [1734979634519, 93044.8, 1734979835053, 92765.6, -0.004009736367791449],
        [1734979856607, 92831.6, 1734980785825, 93110.2, 0.001992153383839735],
        [1735205385386, 95595.6, 1735207318907, 95308.7, -0.004010218374608131],
        [1735219472658, 95386.2, 1735221346378, 95672.9, 0.001996668858161522],
        [1735223766278, 95382.9, 1735224330538, 95669.7, 0.001997814355015227],
        [1735254298503, 95445.4, 1735258300869, 95731.9, 0.0019927328299135247],
        [1735313703428, 94216.0, 1735314845623, 94498.9, 0.001993685640785171],
        [1735319002260, 94105.3, 1735320033325, 93822.7, -0.004012064244580559],
        [1735320066536, 93815.1, 1735320233775, 93533.6, -0.004009613657551924],
    ],
    "high_order": [
        [1733345186350, 98901.7, 1733346796796, 98604.8, 0.002001970643578322],
        [1733355042047, 98995.4, 1733357127181, 98698.7, 0.0019971089565777875],
        [1733365629976, 98912.5, 1733365772342, 99210.3, -0.0040107418172626854],
        [1733365790597, 99311.9, 1733365929045, 99611.0, -0.004011723670577338],
        [1733365934166, 99536.5, 1733365949416, 99836.1, -0.004009951123457366],
        [1733366467231, 100901.7, 1733366470167, 101205.6, -0.004011842218713978],
        [1733367989066, 102908.4, 1733367992432, 103218.1, -0.00400947250175887],
        [1733505578209, 99298.7, 1733506695075, 99597.5, -0.004009102838204436],
        [1733579979655, 99422.1, 1733581046294, 99123.7, 0.002001344771434211],
        [1733830349694, 97684.9, 1733832975024, 97391.7, 0.002001487435622008],
        [1733836366002, 97698.5, 1733839638817, 97992.6, -0.004010281631754985],
        [1733889828578, 97473.2, 1733898679031, 97766.5, -0.004009032226294117],
        [1733920678424, 98438.2, 1733923811307, 98734.5, -0.004010010341513825],
        [1734309242607, 106088.6, 1734309243716, 106407.9, -0.004009748455536032],
        [1734365479547, 106680.3, 1734367069861, 107001.4, -0.004009927793603762],
        [1734367071046, 107051.7, 1734367449068, 106730.3, 0.002002287679691129],
        [1734370378990, 107070.9, 1734373053044, 107393.2, -0.004010154953400068],
        [1734373059393, 107366.2, 1734373393034, 107689.4, -0.004010258349462003],
        [1734609247261, 102342.6, 1734611933451, 102035.5, 0.002000705473576115],
        [1734617163584, 102163.5, 1734618674021, 102471.0, -0.004009881219809385],
        [1734707176378, 96123.3, 1734707469197, 95835.8, 0.001990950165048467],
        [1734717403058, 97272.1, 1734718118885, 96980.3, 0.0019998324288259548],
        [1734720555769, 97107.8, 1734721364052, 96816.1, 0.0020038781642669487],
        [1734751891931, 97353.4, 1734757878946, 97647.0, -0.004015816602193633],
        [1734759646579, 97761.8, 1734759992162, 98056.0, -0.004009355392392443],
        [1734760548206, 98397.1, 1734761412580, 98693.4, -0.004011267608496571],
        [1734996243613, 95391.7, 1734996490676, 95106.5, 0.0019897779366547805],
        [1735122021003, 99119.1, 1735122433191, 98821.7, 0.002000430794872111],
    ],
}

low_returns = [trade[-1] for trade in data["low_order"]]
high_returns = [trade[-1] for trade in data["high_order"]]
print(len(low_returns) + len(high_returns))

# all_returns = low_returns + high_returns
# total_return = sum(all_returns)
# print(total_return)

# winning_trades = len([r for r in all_returns if r > 0])
# total_trades = len(all_returns)
# win_rate = winning_trades / total_trades
# print(f"胜率: {win_rate:.2%}")

# Calculate cumulative returns
cumulative_low_returns = np.cumprod(1 + np.array(low_returns)) - 1
cumulative_high_returns = np.cumprod(1 + np.array(high_returns)) - 1
print(
    "total_result: ",
    cumulative_low_returns[-1]
    + cumulative_high_returns[-1],
)

# Calculate individual returns
individual_low_returns = np.array(low_returns)
individual_high_returns = np.array(high_returns)

low_order_times = [trade[0] for trade in data["low_order"]]
high_order_times = [trade[0] for trade in data["high_order"]]
plt.figure(figsize=(12, 12))
plt.subplot(2, 1, 1)
plt.plot(
    low_order_times,
    cumulative_low_returns,
    label="Cumulative Low Returns",
    color="blue",
)
plt.plot(
    high_order_times,
    cumulative_high_returns,
    label="Cumulative High Returns",
    color="orange",
)
plt.scatter(
    low_order_times,
    individual_low_returns,
    color="blue",
    alpha=0.5,
    label="Individual Low Returns",
    s=10,
)
plt.scatter(
    high_order_times,
    individual_high_returns,
    color="orange",
    alpha=0.5,
    label="Individual High Returns",
    s=10,
)
plt.title("Cumulative and Individual Returns Visualization")
plt.xlabel("Time")
plt.ylabel("Return")
plt.legend()
plt.grid()


# Define a time threshold (in milliseconds, for example)
time_threshold = 1000 * 3 * 60  # 1 second

# Filter low order times and returns
filtered_low_order_times = []
filtered_low_returns = []

last_time = None
for time, return_value in zip(low_order_times, individual_low_returns):
    if last_time is None or (time - last_time) > time_threshold:
        filtered_low_order_times.append(time)
        filtered_low_returns.append(return_value)
        last_time = time

# Filter high order times and returns
filtered_high_order_times = []
filtered_high_returns = []

last_time = None
for time, return_value in zip(high_order_times, individual_high_returns):
    if last_time is None or (time - last_time) > time_threshold:
        filtered_high_order_times.append(time)
        filtered_high_returns.append(return_value)
        last_time = time

# Print the count of filtered high returns
print(
    f"Filtered Returns Count: {len(filtered_high_returns) + len(filtered_low_returns)}"
)

# Plotting
# plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 2)
plt.plot(
    filtered_low_order_times,
    np.cumprod(1 + np.array(filtered_low_returns)) - 1,
    label="Cumulative Low Returns",
    color="blue",
)
plt.plot(
    filtered_high_order_times,
    np.cumprod(1 + np.array(filtered_high_returns)) - 1,
    label="Cumulative High Returns",
    color="orange",
)
plt.scatter(
    filtered_low_order_times,
    filtered_low_returns,
    color="blue",
    alpha=0.5,
    label="Individual Low Returns",
    s=10,
)
plt.scatter(
    filtered_high_order_times,
    filtered_high_returns,
    color="orange",
    alpha=0.5,
    label="Individual High Returns",
    s=10,
)
plt.title("Cumulative and Individual Returns Visualization")
plt.xlabel("Time")
plt.ylabel("Return")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()


low_order_durations = [trade[2] - trade[0] for trade in data["low_order"]]
high_order_durations = [trade[2] - trade[0] for trade in data["high_order"]]

# 绘制开仓到平仓时间分布图
plt.figure(figsize=(12, 6))
plt.hist(
    low_order_durations,
    bins=50,
    alpha=0.5,
    label="Low Order Durations",
    color="blue",
    density=True,
)
plt.hist(
    high_order_durations,
    bins=50,
    alpha=0.5,
    label="High Order Durations",
    color="orange",
    density=True,
)
plt.title("Distribution of Trade Durations")
plt.xlabel("Duration (milliseconds)")
plt.ylabel("Density")
plt.legend()
plt.grid()
plt.show()

import os
import csv
import datetime

class TradingStrategy:
    def __init__(self, data, trend_high, trend_low, basic_config, trend_config, trading_config, record_trade, result_trade, total_result):
        self.data = data
        self.trend_high = trend_high
        self.trend_low = trend_low
        self.basic_config = basic_config
        self.trend_config = trend_config
        self.trading_config = trading_config
        
        self.record_trade = record_trade#开仓记录
        self.result_trade = result_trade#收益记录
        self.total_result = total_result
        
        self.coin_type = self.basic_config['coin_type']
        
        self.delay = self.trend_config['delay'] - 1
        
        self.high_threshold = self.trading_config['high_threshold']
        self.low_threshold = self.trading_config['low_threshold']
        self.number = self.trading_config['number']
        self.reverse_percent = self.trading_config['reverse_percent']
        self.take_profit = self.trading_config['take_profit']
        self.stop_loss = self.trading_config['stop_loss']
        

        self.new_data = self.data[len(self.trend_high) + self.delay]
        self.current_time = self.new_data[0]
        self.high_point = []
        self.low_point = []
        self.trend_to_point()
        if self.record_trade == []:
            self.judgement()#没单才开仓
        if self.record_trade != []:
            self.take()#有单才平仓
        
        self.return_data()
        
        
    def take(self):
        new_record_trade = []
        for trade in self.record_trade:
            if trade[-1] == 'long':#看跌
                if 1 - trade[0][1] / self.new_data[1] > self.stop_loss:#新价格比开仓价格高且大于止损
                    print('get loss')
                    self.result_trade.append([self.new_data, 'short take'])
                    self.total_result[1]+=1
                elif 1 - self.new_data[2] / trade[0][1] > self.take_profit:#新价格比开仓价格低且大于止盈
                    print('get profit')
                    self.result_trade.append([self.new_data, 'short take'])
                    self.total_result[0]+=1
                else:
                    new_record_trade.append(trade)
            elif trade[-1] == 'short':#看涨
                if 1 - self.new_data[2] / trade[0][1] > self.stop_loss:#新价格比开仓价格低且大于止损
                    print('get loss')
                    self.result_trade.append([self.new_data, 'long take'])
                    self.total_result[1]+=1
                elif 1 - trade[0][1] / self.new_data[1] > self.take_profit:#新价格比开仓价格高且大于止盈
                    print('get profit')
                    self.result_trade.append([self.new_data, 'long take'])
                    self.total_result[0]+=1
                else:
                    new_record_trade.append(trade)
        self.record_trade = new_record_trade
        print(f'total result:{self.total_result}')
        if self.total_result[0]+self.total_result[1] != 0:
            print(f'current win rate:{self.total_result[0]/(self.total_result[0]+self.total_result[1])}')

    def trend_to_point(self):
        for i in range(len(self.trend_high)):
            for slope, j in self.trend_high[i]:
                self.high_point.append(self.data[j][1] + slope * (self.current_time - self.data[j][0]))
        for i in range(len(self.trend_low)):
            for slope, j in self.trend_low[i]:
                self.low_point.append(self.data[j][2] + slope * (self.current_time - self.data[j][0]))
                
    def judgement(self):
        
        high_count = sum(
            1 for hp in self.high_point
            if abs(1 - self.new_data[1] / hp) < self.high_threshold
        )#至少要有一定数量的密集趋势线且接近价格它们
        if high_count >= self.number:
            if 1 - max(self.low_point) / self.new_data[1] > self.reverse_percent:#至少大于潜在利润
                print(high_count)
                self.buy()
                
        low_count = sum(
            1 for lp in self.low_point
            if abs(1 - lp / self.new_data[2]) < self.low_threshold
        )
        if low_count >= self.number:
            if 1 - self.new_data[1] / min(self.high_point) > self.reverse_percent:
                print(low_count)
                self.sell()
        
    
    def buy(self):
        print(1 - self.new_data[1] / min(self.high_point))
        self.record_trade.append([self.new_data, 'long'])
    
    def sell(self):
        print(1 - max(self.low_point) / self.new_data[2])
        self.record_trade.append([self.new_data, 'short'])
        
    def return_data(self):
        return self.record_trade

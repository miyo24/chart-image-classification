import pandas as pd
import mplfinance as mpf
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import datetime as dt
import os
import sys


# year month day
args = sys.argv[1:]
year, month, day = map(int, args[0].split("-"))
csv_file = f"download/{year:02}-{month:02}-{day:02}.csv"
print(csv_file)

df = pd.read_csv(csv_file, parse_dates=["timestamp"])

start_ts = dt.datetime(year,month,day,0,0)
end_ts = dt.datetime(year,month,day,0,59,59)
finish_ts = dt.datetime(year,month,day,23,0)
data_num = 0
position_list = [0] * 1321
win = 'win'
lose = 'lose'
timeup = 'timeup'
result_list = [''] * 1321
win_range = 0.2
lose_range = 0.1
candle_dir = 'candle_data'
prefix = f'{year}-{month}-{day}_'
extension = '.png'

i = 0
for timestamp in df["timestamp"]:
  if timestamp > end_ts and timestamp <= finish_ts:
    tick_data = df[(df["timestamp"] >= start_ts) & (df["timestamp"] <= end_ts)]
    print("timestamp", timestamp)
    print("start_ts", start_ts)
    print("end_ts", end_ts)
    print("data_num", data_num)
    print("i", i)
    print(tick_data)
    # 1分足に変換（OHLCデータを作成）
    tick_data["minute"] = tick_data["timestamp"].dt.floor("min")
    ohlc = tick_data.groupby("minute")["askPrice"].agg(["first", "max", "min", "last"])
    ohlc.columns = ["open", "high", "low", "close"]
    
    # インデックスを datetime に変換
    ohlc.index = pd.to_datetime(ohlc.index)
    
    # ローソク足チャートを描画
    myrcparams={"xtick.labelsize": 3}
    mpf_style = mpf.make_mpf_style(base_mpl_style="ggplot",rc=myrcparams)
    fig, ax = plt.subplots(figsize=(3, 3))  # 画像サイズ
    
    # 価格をそのまま表示するためにフォーマットを設定
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.2f}".split(".")[1]))
    ax.set_position([0.15, 0.3, 0.75, 0.6])
    ax.set_ylabel('')
#    ax.tick_params(axis='x', labelsize=4)
#    ax.tick_params(axis='y', labelsize=4)
    mpf.plot(ohlc, type="candle", style=mpf_style, ax=ax, ylabel="",tight_layout=True)
    
    # 画像として保存
#    ts_str = old_ts.strftime('%Y-%m-%d_%H%M%S')
    plt.savefig(f"{candle_dir}/{prefix}{data_num}.png", dpi=100)
#    plt.show()
#    print("ローソク足画像を生成しました: candlestick_chart.png")

    position_num = tick_data.tail(1).index[0] + 1
    position_list[data_num] = df.at[position_num, "askPrice"]
    start_ts += dt.timedelta(minutes=1) 
    end_ts += dt.timedelta(minutes=1) 
    data_num += 1
  now_price = df.at[i, 'askPrice']
  i += 1
  j = 0
  for position in position_list:
#    print("position", position)
#    print("now_price", now_price)
#    print("j", j)
    if position == 0 or result_list[j] != '':
#      print("continue")
      j += 1
      continue
    if now_price >= position + win_range:
#      print("win!" , str(now_price + win_range))
#      print("now_price", now_price)
#      print("timestamp", timestamp)
#      print("position", position)
#      print("j", j)
      result_list[j] = win
    elif now_price <= position - lose_range:
#      print("lose!", str(now_price - lose_range))
#      print("now_price", now_price)
#      print("timestamp", timestamp)
#      print("position", position)
#      print("j", j)
      result_list[j] = lose
    j += 1

print("position--------------")
print(position_list)
print('\n\nresult--------------')
print(result_list)

i = 0
for result in result_list:
  if result == win:
    os.rename(f"{candle_dir}/{prefix}{i}{extension}", f"{candle_dir}/{win}/{prefix}{i}{extension}")
  elif result == lose:
    os.rename(f"{candle_dir}/{prefix}{i}{extension}", f"{candle_dir}/{lose}/{prefix}{i}{extension}")
  else:
    os.rename(f"{candle_dir}/{prefix}{i}{extension}", f"{candle_dir}/{timeup}/{prefix}{i}{extension}")
  i += 1
  

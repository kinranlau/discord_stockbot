#%%
import yfinance as yf

import matplotlib.pyplot as plt
import pandas as pd
#%%
# set to a non-interactive backend because we will run matplotlib in a separate thread
# otherwise you will get a warning: 'user warning starting a matplotlib gui outside of the main thread will likely fail'
import matplotlib
matplotlib.use('Agg')  
####################################################################################################
#%%
# Parameters for indicators

# Stochastic Oscillator
# standard overbought and oversold thresholds
stoc_overbought_threshold = 80
stoc_oversold_threshold = 20

# RSI
# standard overbought and oversold thresholds
rsi_overbought_threshold = 70
rsi_oversold_threshold = 30

# MACD
# standard MACD threshold
MACD_threshold = 0
####################################################################################################
#%%
def analyze_stock(): 
    # import watchlist
    watchlist = pd.read_csv('#nasdaq100.csv')

    # define tickers
    tickers = watchlist['Ticker'].tolist()

    # dictionary for mapping ticker to company name
    ticker_dict = {}
    for index, row in watchlist.iterrows():
        ticker_dict[row['Ticker']] = row['Company']
    ####################################################################################################
    #%%
    # lists to store the results
    watchlist_buy_ticker = []
    watchlist_buy_name = []

    counter = 1

    for ticker in tickers:
        # escape IndexError
        try:
            data = yf.Ticker(ticker)
            # get the data
            # 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
            time = '2y'
            df = data.history(period=time)

            # inverse the dataframe in choronological order
            df = df.sort_values(by='Date', ascending=True)

            # default dataframe has 'Date' as the index, reset to make 'Date' a column
            df = df.reset_index()
            ############################################################
            # Bollinger Bands
            ############################################################
            # Define the Bollinger Bands period
            bollinger_period = 20
            # Define the number of standard deviations to use for the upper and lower bands
            bollinger_std = 2

            # typical price = (high + low + close) / 3
            df['Typical price'] = (df["High"] + df["Low"] + df["Close"]) / 3
            # calculate the moving average of the typical price
            df['Moving average'] = df['Typical price'].rolling(bollinger_period).mean()
            # calculate the moving standard deviation of the typical price
            # note that the ddof parameter is set to 0 instead of the default 1, assuming we are calculating the population standard deviation
            df['Moving std'] = df['Typical price'].rolling(bollinger_period).std(ddof=0)

            # calculate the upper and lower bands
            df['Upper band'] = df['Moving average'] + (df['Moving std'] * bollinger_std)
            df['Lower band'] = df['Moving average'] - (df['Moving std'] * bollinger_std)
            ############################################################
            # Stochastic Oscillator
            ############################################################
            # define stochastic oscillator parameters
            k_period = 14
            d_period = 3

            # calculate the highest high and lowest low in the past k_period days
            df["Highest high"] = df["High"].rolling(k_period).max()
            df["Lowest low"] = df["Low"].rolling(k_period).min()

            # calculate the stochastic oscillator %K
            df["%K"] = (df["Close"] - df["Lowest low"]) / (df["Highest high"] - df["Lowest low"]) * 100

            # calculate the %D, which is the 3-day SMA of %K
            df["%D"] = df["%K"].rolling(d_period).mean()

            # calculate the difference between current and previous %D
            df["%D diff"] = df["%D"].diff()

            # fill the first days with the first available value for better visualization (align the axes)
            df.loc[df.index[:k_period], '%K'] = df['%K'].iloc[k_period]
            df.loc[df.index[:k_period+d_period], '%D'] = df['%D'].iloc[k_period+d_period]
            ############################################################
            # RSI
            ############################################################
            # Define the RSI period
            rsi_period = 14
            # calculate moving average of RSI similar to stochastic oscillator %D to smoothen out the RSI a bit
            rsi_MA_period = 3

            # Calculate the price change for each period
            # DataFrame.diff() function gets the difference between the current element and its previous element (the one above the current one)
            # Note that close price was used here, but you can specify "Open", "High", "Low", "Close"
            price_change = df["Close"].diff()

            # Calculate the upward price movements (positive price changes)
            upward_price = price_change.where(price_change > 0, 0)

            # Calculate the downward price movements (negative price changes)
            downward_price = -price_change.where(price_change < 0, 0)

            # Calculate the average gain and loss for the RSI period
            # DataFrame.rolling() gives you the rolling window
            average_gain = upward_price.rolling(rsi_period).mean()
            average_loss = downward_price.rolling(rsi_period).mean()

            # Calculate the relative strength (RS)
            rs = average_gain / average_loss

            # Calculate the RSI using the RS
            rsi = 100 - (100 / (1 + rs))

            # add the RSI to the DataFrame
            df["RSI"] = rsi

            # calculate moving average of RSI
            df['RSI_MA'] = df['RSI'].rolling(rsi_MA_period).mean()

            # calculate difference between current and previous RSI moving average
            df['RSI_MA_diff'] = df['RSI_MA'].diff()

            # the initial RSI period is NaN, so we need to pad it with the first RSI value for alignment (better visualization)
            df.loc[df.index[:rsi_period], 'RSI'] = df['RSI'].iloc[rsi_period]
            ############################################################
            # MACD
            ############################################################
            # Define the short-term and long-term moving average periods
            short_ma_period = 12
            long_ma_period = 26
            # Define the signal line period
            signal_period = 9
            # calculate moving average of MACD histogram similar to stochastic oscillator %D to smoothen out the MACD histogram a bit
            MACD_MA_period = 3

            # Calculate the short-term and long-term exponential moving averages (EMA)
            short_ema = df["Close"].ewm(span=short_ma_period, adjust=False).mean()
            long_ema = df["Close"].ewm(span=long_ma_period, adjust=False).mean()

            # Calculate the MACD line as the difference between the short-term and long-term EMAs
            macd_line = short_ema - long_ema

            # Calculate the signal line as the 9-day EMA of the MACD line
            signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()

            # Calculate the MACD histogram as the difference between the MACD line and signal line
            macd_histogram = macd_line - signal_line

            # add the MACD to the DataFrame
            df["MACD"] = macd_line
            df["MACD signal line"] = signal_line
            df["MACD histogram"] = macd_histogram

            # calculate moving average of MACD histogram
            df['MACD_MA'] = df['MACD histogram'].rolling(MACD_MA_period).mean()

            # calculate difference between current and previous MACD histogram moving average
            df['MACD_MA_diff'] = df['MACD_MA'].diff()
            ############################################################
            # flags and lists for picking index
            ############################################################
            # Stochastic Oscillator
            ############################################################
            stoc_overbought_flag = False
            stoc_oversold_flag = False

            stoc_up = []
            stoc_down = []
            ############################################################
            # RSI
            ############################################################
            rsi_overbought_flag = False
            rsi_oversold_flag = False

            rsi_up = []
            rsi_down = []
            ############################################################
            # MACD
            ############################################################
            MACD_overbought_flag = False
            MACD_oversold_flag = False
            
            MACD_up = []
            MACD_down = []
            ############################################################
            # Overall up alert: when all signals are up (max 1 + 1 + 1 = 3)

            # initialize the signals
            stoc_signal = 0
            rsi_signal = 0
            macd_signal = 0
            df["Overall signal"] = 0

            # list to store index when the overall signal first reaches 3
            overall_up = []
            ############################################################
            # step through the dataframe
            for index, row in df.iterrows():
                ############################################################
                # Stochastic Oscillator
                ############################################################          
                # step through the stochastic oscillator %D diff
                # as long as %D is above the overbought threshold (e.g. 80), keep checking the diff
                # if the diff is positive, turn the overbought flag to True
                # if the diff is negative, turn the overbought flag to False
                # make sure we didn't miss the point when %D first drop below the overbought threshold
                if row['%D'] > stoc_overbought_threshold and row['%D diff'] > 0 and not stoc_overbought_flag:
                    stoc_overbought_flag = True
                elif row['%D'] > stoc_overbought_threshold and row['%D diff'] < 0 and stoc_overbought_flag:
                    stoc_overbought_flag = False
                    stoc_down.append(index)
                    stoc_signal = 0
                elif row['%D'] < stoc_overbought_threshold and stoc_overbought_flag:
                    stoc_overbought_flag = False
                    stoc_down.append(index)
                    stoc_signal = 0

                # step through the stochastic oscillator %D diff
                # as long as %D is below the oversold threshold (e.g. 20), keep checking the diff
                # if the diff is negative, turn the oversold flag to True
                # if the diff is positive, turn the oversold flag to False
                # make sure we didn't miss the point when %D first rise above the oversold threshold
                if row['%D'] < stoc_oversold_threshold and row['%D diff'] < 0 and not stoc_oversold_flag:
                    stoc_oversold_flag = True
                    stoc_down.append(index)
                    stoc_signal = 0
                elif row['%D'] < stoc_oversold_threshold and row['%D diff'] > 0 and stoc_oversold_flag:
                    stoc_oversold_flag = False
                    stoc_up.append(index)
                    stoc_signal = 1
                elif row['%D'] > stoc_oversold_threshold and stoc_oversold_flag:
                    stoc_oversold_flag = False
                    stoc_up.append(index)
                    stoc_signal = 1
                ############################################################
                # RSI
                ############################################################
                # step through the RSI
                # as long as RSI is above the overbought threshold (e.g. 70), keep checking the diff
                # if the diff is positive, turn the overbought flag to True
                # if the diff is negative, turn the overbought flag to False
                # make sure we didn't miss the point when RSI first drop below the overbought threshold
                if row['RSI'] > rsi_overbought_threshold and row['RSI_MA_diff'] > 0 and not rsi_overbought_flag:
                    rsi_overbought_flag = True
                elif row['RSI'] > rsi_overbought_threshold and row['RSI_MA_diff'] < 0 and rsi_overbought_flag:
                    rsi_overbought_flag = False
                    rsi_down.append(index)
                    rsi_signal = 0
                elif row['RSI'] < rsi_overbought_threshold and rsi_overbought_flag:
                    rsi_overbought_flag = False
                    rsi_down.append(index)
                    rsi_signal = 0

                # step through the RSI
                # as long as RSI is below the oversold threshold (e.g. 30), keep checking the diff
                # if the diff is negative, turn the oversold flag to True
                # if the diff is positive, turn the oversold flag to False
                # make sure we didn't miss the point when RSI first rise above the oversold threshold
                if row['RSI'] < rsi_oversold_threshold and row['RSI_MA_diff'] < 0 and not rsi_oversold_flag:
                    rsi_oversold_flag = True
                    rsi_down.append(index)
                    rsi_signal = 0
                elif row['RSI'] < rsi_oversold_threshold and row['RSI_MA_diff'] > 0 and rsi_oversold_flag:
                    rsi_oversold_flag = False
                    rsi_up.append(index)
                    rsi_signal = 1
                elif row['RSI'] > rsi_oversold_threshold and rsi_oversold_flag:
                    rsi_oversold_flag = False
                    rsi_up.append(index)
                    rsi_signal = 1
                ############################################################
                # MACD
                ############################################################
                # step through the MACD histogram
                # as long as MACD histogram is above the overbought threshold (e.g. 0), keep checking the diff
                # if the diff is positive, turn the overbought flag to True
                # if the diff is negative, turn the overbought flag to False
                # make sure we didn't miss the point when MACD histogram first drop below the overbought threshold
                if row['MACD histogram'] > MACD_threshold and row['MACD_MA_diff'] > 0 and not MACD_overbought_flag:
                    MACD_overbought_flag = True
                elif row['MACD histogram'] > MACD_threshold and row['MACD_MA_diff'] < 0 and MACD_overbought_flag:
                    MACD_overbought_flag = False
                    MACD_down.append(index)
                    macd_signal = 0
                elif row['MACD histogram'] < MACD_threshold and MACD_overbought_flag:
                    MACD_overbought_flag = False
                    MACD_down.append(index)
                    macd_signal = 0

                # step through the MACD histogram
                # as long as MACD histogram is below the oversold threshold (e.g. 0), keep checking the diff
                # if the diff is negative, turn the oversold flag to True
                # if the diff is positive, turn the oversold flag to False
                # make sure we didn't miss the point when MACD histogram first rise above the oversold threshold
                if row['MACD histogram'] < MACD_threshold and row['MACD_MA_diff'] < 0 and not MACD_oversold_flag:
                    MACD_oversold_flag = True
                    MACD_down.append(index)
                    macd_signal = 0
                elif row['MACD histogram'] < MACD_threshold and row['RSI_MA_diff'] > 0 and MACD_oversold_flag:
                    MACD_oversold_flag = False
                    MACD_up.append(index)
                    macd_signal = 1
                elif row['MACD histogram'] > MACD_threshold and MACD_oversold_flag:
                    MACD_oversold_flag = False
                    MACD_up.append(index)
                    macd_signal = 1
                ############################################################
                # Overall signal
                ############################################################
                overall_signal = stoc_signal + rsi_signal + macd_signal
                # update the overall signal
                df.loc[index, 'Overall signal'] = overall_signal

                # only add to the list if this is the first time changing to 3
                if overall_signal == 3:
                    overall_last = df['Overall signal'].loc[index-1]
                    if overall_last < 3:
                        overall_up.append(index)
            ############################################################
            if df['Overall signal'].iloc[-1] == 3:
                 # if overall signal equals max, append it to the watchlist
                watchlist_buy_ticker.append(ticker)
                
                # convert ticker to company name
                company_name = ticker_dict[ticker]
                watchlist_buy_name.append(company_name)
                ############################################################
                # plot figure if the overall signal is the highest
                # plot stock price with indicators from stochastic oscillator, RSI, and MACD
                # 1st graph: stock price
                # 2nd graph: stochastic oscillator
                # 3rd graph: RSI
                # 4th graph: MACD
                fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1)
                # set the size of the figure
                fig.set_size_inches(12.5, 12)

                # 1st graph: stock price
                ax1.plot(df['Date'], df["Close"], label="Price", color = 'k')
                # also label the right y-axis with price
                ax1.yaxis.set_ticks_position('both')
                ax1.yaxis.set_tick_params(labelright=True)

                ax1.set_title(ticker + ": " + company_name)
                ax1.set_ylabel("Price")

                # plot bollinger bands
                bollinger_color = 'darkorange'
                ax1.plot(df['Date'], df['Moving average'], label="Bollinger bands", color = bollinger_color, linestyle = '--')
                ax1.plot(df['Date'], df["Upper band"], color = bollinger_color, linestyle = '--')
                ax1.plot(df['Date'], df["Lower band"], color = bollinger_color, linestyle = '--')
                ax1.fill_between(df['Date'], df["Upper band"], df["Lower band"], color = bollinger_color, alpha = 0.1)

                # overall up signal as vertical lines
                ax1.axvline(df['Date'].loc[overall_up[0]], color='g', linestyle='--', label='Overall up alert')
                for index in overall_up[1:]:
                    ax1.axvline(x=df['Date'].loc[index], color='g', linestyle='--')

                # plot RSI, MACD, stochastic signals on top of the stock price as scatter plot
                size = 25
        
                # RSI signal
                RSI_marker = 'x'
                ax1.scatter(df['Date'].loc[rsi_up], df.loc[rsi_up]["Close"], color='g', marker=RSI_marker, s = size)
                ax1.scatter(df['Date'].loc[rsi_down], df.loc[rsi_down]["Close"], color='r', marker=RSI_marker, s = size)
                # MACD signal
                MACD_marker = 'o'
                ax1.scatter(df['Date'].loc[MACD_up], df.loc[MACD_up]["Close"], color='g', marker=MACD_marker, s = size)
                ax1.scatter(df['Date'].loc[MACD_down], df.loc[MACD_down]["Close"], color='r', marker=MACD_marker, s = size)
                # stochastic signal
                stoc_marker = '^'
                ax1.scatter(df['Date'].loc[stoc_up], df.loc[stoc_up]["Close"], color='g', marker=stoc_marker, s = size)
                ax1.scatter(df['Date'].loc[stoc_down], df.loc[stoc_down]["Close"], color='r', marker=stoc_marker, s = size)

                ax1.legend(loc="upper left")
                ###########################################################
                # 2nd graph: stochastic oscillator
                ax2.plot(df['Date'], df["%K"], label="%K", color = 'k')
                ax2.plot(df['Date'], df["%D"], label="%D", color = 'navy', linestyle='--')
                ax2.set_ylabel("%K, %D")

                # also label the stochastic oscillator with the stochastic up and down points
                ax2.scatter(df['Date'].loc[stoc_up], df.loc[stoc_up]["%K"], color='g', marker=stoc_marker, label='Stochastic up')
                ax2.scatter(df['Date'].loc[stoc_down], df.loc[stoc_down]["%K"], color='r', marker=stoc_marker, label='Stochastic down')

                # plot the oversold and overbought lines
                ax2.axhline(stoc_oversold_threshold, color='g', linestyle='--')
                ax2.axhline(stoc_overbought_threshold, color='r', linestyle='--')

                ax2.legend(loc="upper left")
                ###########################################################
                # 3rd graph: RSI
                ax3.plot(df['Date'], df["RSI"], label="RSI", color = 'k')
                ax3.set_ylabel("RSI")

                # also label the RSI with the RSI up and down points
                ax3.scatter(df['Date'].loc[rsi_up], df.loc[rsi_up]["RSI"], color='g', marker=RSI_marker, label='RSI up')
                ax3.scatter(df['Date'].loc[rsi_down], df.loc[rsi_down]["RSI"], color='r', marker=RSI_marker, label='RSI down')

                # plot the oversold and overbought lines
                ax3.axhline(rsi_oversold_threshold, color='g', linestyle='--')
                ax3.axhline(rsi_overbought_threshold, color='r', linestyle='--')

                ax3.legend(loc="upper left")
                ###########################################################
                # 4th graph: MACD
                # plot the MACD and signal line
                ax4.plot(df['Date'], df["MACD"], label="MACD", color = 'k')
                ax4.plot(df['Date'], df["MACD signal line"], label="Signal line", color = 'navy', linestyle='--')
                ax4.set_ylabel("MACD")

                # also label the MACD with the MACD up and down points
                ax4.scatter(df['Date'].loc[MACD_up], df.loc[MACD_up]["MACD histogram"], color='g', marker=MACD_marker, label='MACD up')
                ax4.scatter(df['Date'].loc[MACD_down], df.loc[MACD_down]["MACD histogram"], color='r', marker=MACD_marker, label='MACD down')

                # plot the MACD histogram, green for positive, red for negative
                ax4.bar(df['Date'], df["MACD histogram"], color=df["MACD histogram"].apply(lambda x: 'g' if x > 0 else 'r'))
                ax4.set_xlabel("Date")

                ax4.legend(loc="upper left")
                ###########################################################
                plt.savefig("image/" + ticker + ".png", dpi=200, facecolor='white')
                plt.close()
            ###########################################################
            # print progress
            print(f'Currently processing {counter} out of {len(tickers)} stocks.')
            counter += 1

        # escape IndexError
        except(IndexError):
            pass

    #%%
    return watchlist_buy_ticker, watchlist_buy_name
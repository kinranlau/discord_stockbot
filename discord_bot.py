#%%
import os
from dotenv import load_dotenv

import asyncio

import discord
import stock_analysis
####################################################################################################
#%%
# get discord token and channel id from .env file
load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')
channel_id = int(os.getenv('DISCORD_CHANNEL_ID'))

# now need to pass intents to client
intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)
####################################################################################################
# event when the bot is ready
@client.event
async def on_ready():
    # print message to console when bot is ready
    print(f'{client.user} has connected to Discord!')
    
    # Call the send_stock_recommendation function and await its result
    await send_stock_recommendation(channel_id)

    # close the bot after sending the message
    await client.close()
####################################################################################################
# send stock recommendation to discord
@client.event
async def send_stock_recommendation(channel_id):
    # Run the stock analysis function in a separate thread using asyncio's run_in_executor()
    # Call `stock_analysis.py` to analyze the stocks
    loop = asyncio.get_event_loop()
    watchlist_buy_ticker, watchlist_buy_name = await loop.run_in_executor(None, stock_analysis.analyze_stock)

    # combine names and tickers in a single list
    watchlist = []
    for name, ticker in zip(watchlist_buy_name, watchlist_buy_ticker):
        watchlist.append(f'{name}({ticker})')

    # get the channel object
    channel = client.get_channel(channel_id) 
    ####################################################################################################
    # discord only allows 10 files to be sent at a time, so we send them out in batches

    # determine how many times to loop
    num_loops = len(watchlist) // 10

    for i in range(num_loops):
        # first prepare the text message
        message = f'You might want to consider these stocks from the watchlist: {watchlist[i*10:(i+1)*10]}.'
        # send the text message
        await channel.send(message)

        # then send the png files
        png_file_list = ['image/' + ticker + '.png' for ticker in watchlist_buy_ticker[i*10:(i+1)*10]]
        # Create a list of Discord file objects from the png files
        files = [discord.File(file) for file in png_file_list]
        # send the files
        await channel.send(files=files)

    # send the remaining files
    # text message
    message = f'You might want to consider these stocks from the watchlist: {watchlist[num_loops*10:]}.'
    await channel.send(message)
    png_file_list = ['image/' + ticker + '.png' for ticker in watchlist_buy_ticker[num_loops*10:]]
    # Create a list of Discord file objects from the png files
    files = [discord.File(file) for file in png_file_list]
    # send the files
    await channel.send(files=files)
####################################################################################################
#%%
if __name__ == "__main__":
    client.run(TOKEN)
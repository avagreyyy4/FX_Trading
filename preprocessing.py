import pandas as pd

df_second_ten = pd.read_csv("data/EUR_GBP_2015_2005.csv")
df_first_ten = pd.read_csv("data/EUR_GBP_2025_2015.csv")

df_exchanges = pd.concat([df_second_ten, df_first_ten], ignore_index = True)

df_exchanges['Close'] = df_exchanges['Price']
df_exchanges = df_exchanges.drop(columns = ['Price', 'Vol.', 'Change %'])
df_exchanges['Date'] = pd.to_datetime(df_exchanges['Date'], format = '%m/%d/%y')

GBP_GDP = pd.read_csv("data/UKNGDP.csv")

GBP_GDP['Date'] = GBP_GDP['observation_date']
GBP_GDP = GBP_GDP.drop(columns = 'observation_date')
GBP_GDP['Date'] = pd.to_datetime(GBP_GDP['Date'], format='%Y-%m-%d')
#GBP_GDP = GBP_GDP[GBP_GDP['Date'] >= df_exchanges['Date'].min()]
GBP_GDP = GBP_GDP.rename(columns = {'UKNGDP': 'UK_GDP'})

df_exchanges = df_exchanges.sort_values('Date')
GBP_GDP = GBP_GDP.sort_values('Date')

merged = pd.merge_asof(
    df_exchanges,
    GBP_GDP[['Date', 'UK_GDP']],
    on='Date',
    direction='backward'
)

EUR_GDP = pd.read_csv("data/EURORGDP.csv")

EUR_GDP = EUR_GDP.rename(columns = {'EURO GDP': 'EURO_GDP',
                                   'observation_date': 'Date'})

EUR_GDP['Date'] = pd.to_datetime(EUR_GDP['Date'], format = '%m/%d/%y')
#EUR_GDP = EUR_GDP[EUR_GDP['Date'] >= df_exchanges['Date'].min()]
EUR_GDP = EUR_GDP.sort_values('Date')

merged = pd.merge_asof(
    merged,
    EUR_GDP[['Date', 'EURO_GDP']],
    on='Date',
    direction='backward'
)

EUR_BoP = pd.read_csv('data/EURO_BoP.csv')

EUR_BoP = EUR_BoP.rename(columns = {'Current account balance, calendar and seasonally adj (BPS.M.Y.I9.W1.S1.S1.T.B.CA._Z._Z._Z.EUR._T._X.N.ALL)': 'EURO_Balance_Payments',
                                   'DATE':'Date'})

EUR_BoP = EUR_BoP.drop(columns = 'TIME PERIOD')
EUR_BoP['Date'] = pd.to_datetime(EUR_BoP['Date'], format='%Y-%m-%d')
EUR_BoP = EUR_BoP.sort_values('Date')


merged = pd.merge_asof(
    merged,
    EUR_BoP[['Date', 'EURO_Balance_Payments']],
    on='Date',
    direction='backward'
)

GBP_BoP = pd.read_csv('data/UkBoP.csv')

GBP_BoP = GBP_BoP.iloc[86:]
GBP_BoP = GBP_BoP.rename(
    columns = {'Title':'Date',
               'BoP Current Account Balance SA £m': 'GBP_Balance_Payments'})

def quarter_to_date(qstr):
    year, quarter = qstr.split()
    q = int(quarter[1])  
    month = {1: 1, 2: 4, 3: 7, 4: 10}[q]
    return pd.Timestamp(year=int(year), month=month, day=1)

GBP_BoP['Date'] = GBP_BoP['Date'].apply(quarter_to_date)

GBP_BoP = GBP_BoP.sort_values('Date')

merged = pd.merge_asof(
    merged,
    GBP_BoP[['Date', 'GBP_Balance_Payments']],
    on='Date',
    direction='backward'
)

UK_trade_balance = pd.read_csv('data/UK_tradeBalance 2.csv')
UK_trade_balance = UK_trade_balance[UK_trade_balance['Direction'] == 'Balance']
UK_trade_balance = UK_trade_balance.drop(columns = ['Series', 'Direction', 'Commodity', 'Area', 'Series ID'])
UK_trade_balance = UK_trade_balance.melt(
    var_name='Date',
    value_name='GBP_trade_balance'
)
UK_trade_balance['Date'] = pd.to_datetime(UK_trade_balance['Date'], format='%Y %b')
UK_trade_balance = UK_trade_balance.sort_values('Date')

merged = pd.merge_asof(
    merged,
    UK_trade_balance[['Date', 'GBP_trade_balance']],
    on='Date',
    direction='backward'
)

EURO_trade_balance = pd.read_csv('data/EuroTradeData.csv')

EURO_trade_balance = EURO_trade_balance[['TIME_PERIOD', 'OBS_VALUE']]
EURO_trade_balance = EURO_trade_balance.rename(columns = {'TIME_PERIOD':'Date', 'OBS_VALUE':'EUR_trade_balance'})
EURO_trade_balance['Date'] = pd.to_datetime(EURO_trade_balance['Date'], format='%Y-%m')
EURO_trade_balance = EURO_trade_balance.sort_values('Date')

merged = pd.merge_asof(
    merged,
    EURO_trade_balance[['Date', 'EUR_trade_balance']],
    on='Date',
    direction='backward'
)

merged.to_csv("data/final_clean_FX_data.csv", index=False)

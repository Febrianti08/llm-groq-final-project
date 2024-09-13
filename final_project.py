import os
import json
import requests
import pandas as pd
from datetime import date, timedelta
from collections import defaultdict
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.tools import tool
from langchain.callbacks import StreamlitCallbackHandler
import streamlit as st
from langchain_community.tools import DuckDuckGoSearchRun

# load_dotenv() # load your .env file
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# SECTORS_API_KEY = os.getenv("SECTORS_API_KEY")
# CALENDAR_API_KEY = os.getenv("CALENDAR_API_KEY")
GROQ_API_KEY = st.secrets['GROQ_API_KEY']
SECTORS_API_KEY = st.secrets['SECTORS_API_KEY']
CALENDAR_API_KEY = st.secrets['CALENDAR_API_KEY']


def retrieve_from_endpoint(url: str) -> dict:
    """

    Args:
        url (str): URL from the tools that retrieves data based on the user's question.

    Raises:
        SystemExit: The SystemExit exception in Python is used to exit the program. 
        When it is raised, the Python interpreter exits the program and stops execution immediately

    Returns:
        dict: return a dictionary format
    """    
    headers = {"Authorization": SECTORS_API_KEY}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.HTTPError as err:
        raise SystemExit(err)
    return json.dumps(data)

def get_today_date() -> str:
    """Designed to return the current date as a string in the format "YYYY-MM-DD"

    Returns:
        str: return a string format
    """    
    today = date.today()
    return today.strftime("%Y-%m-%d")

def fetch_holidays_and_mass_leave(year: int) -> set:
    cache_file = f"holidays_{year}.json"
    
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            holiday_dates = {pd.Timestamp(date) for date in json.load(f)}
    else:
        url = f"https://calendarific.com/api/v2/holidays?&CALENDAR_API_KEY={CALENDAR_API_KEY}&country=ID&year={year}"
        response = requests.get(url)
        holidays = response.json().get('response', {}).get('holidays', [])
        
        holiday_dates = {pd.Timestamp(holiday['date']['iso']) 
                         for holiday in holidays 
                         if holiday['type'][0] in ['National holiday', 'Public holiday', 'Observance']}
        
        with open(cache_file, 'w') as f:
            json.dump([str(date) for date in holiday_dates], f)
    
    return holiday_dates

@tool
def get_next_workday(start_date: str) -> str:
    """This fuction is to check holidays_and_mass_leave and return workday date

    Args:
        start_date (str): Date string
        holidays (set): Return from fetch_holidays_and_mass_leave function

    Returns:
        date: Workday date
    """
    holidays = fetch_holidays_and_mass_leave(date.today().year)
    current_date = pd.Timestamp(start_date)
    while current_date in holidays or current_date.weekday() >= 5:
        current_date += timedelta(days=1)
    res = {}
    res['date'] = current_date.strftime('%Y-%m-%d')
    return json.dumps(res)


def _get_next_workday_func(start_date: str) -> date:
    """This fuction is to check holidays_and_mass_leave and return workday date

    Args:
        start_date (str): Date string
        holidays (set): Return from fetch_holidays_and_mass_leave function

    Returns:
        date: Workday date
    """
    holidays = fetch_holidays_and_mass_leave(date.today().year)
    current_date = pd.Timestamp(start_date)
    while current_date in holidays or current_date.weekday() >= 5:
        current_date += timedelta(days=1)
    res = {}
    res['date'] = current_date.strftime('%Y-%m-%d')
    return res['date']

def get_sorted_companies_by_total_volume(data: dict) -> list:
    # Dictionary to store the total volume and company name for each symbol
    company_volumes = defaultdict(lambda: {"company_name": "", "total_volume": 0})
    
    # Iterate through each day's data
    for date, companies in data.items():
        for company in companies:
            symbol = company["symbol"]
            volume = company["volume"]
            company_name = company["company_name"]
            
            # Update total volume and set company name
            company_volumes[symbol]["total_volume"] += volume
            company_volumes[symbol]["company_name"] = company_name
    
    # Sort companies by total volume in descending order
    sorted_companies = sorted(company_volumes.items(), key=lambda x: x[1]["total_volume"], reverse=True)
    
    # Prepare the result in a list of dicts
    result = [
        {"symbol": symbol, "company_name": details["company_name"], "total_volume": details["total_volume"]}
        for symbol, details in sorted_companies
    ]
    
    return result

@tool
def get_company_report(ticker: str, section : str) -> str:
    """Fetches the company report for a specified stock ticker, organized by the requested sections.
    This function retrieves a report for a given stock symbol, allowing you to specify the desired sections of the report.
    If the question is about largest market cap then using get_company_report to get the data

    Args:
        ticker (str): The 4-character stock symbol representing the company. This parameter only accepts a single stock symbol.
        section (str): The sections of the company report you wish to retrieve. Multiple sections can be specified, separated by commas.
        Available report Section
            - Overview provides general company data including : listing board, industry, sector, market capitalization, market rank, address, number of employees, listing date, website, contact information, latest close price, and daily price change
            - Use the get_company_report function with the 'financials' parameter for: Tax, revenue, earnings, bebt, assets, equity, liabilities, and other financial metrics
            - Use the get_company_report function with the 'dividend' parameter for: Historical dividends, Annual yield, yield TTM, Average dividend yield, Payout ratio, cash payout ratio, Last ex-dividend date
            - Use the get_company_report function with the 'management' parameter for: Key executives, Executives' shareholdings
            - Use the get_company_report function with the 'ownership' parameter for: Major shareholders, Top transactions, Monthly net transactions
            - Use the get_company_report function with the 'peers' parameter for: Information about peer companies
            - Use the get_company_report function with the 'valuation' parameter for: Latest close price, daily price changes, Forward P/E, price-to-cash flow, Enterprise multiples (enterprise-to-revenue, enterprise-to-EBITDA), PEG ratio, intrinsic value,Historical valuation (P/E, P/B, P/S)

    Returns:
        str: The content of the company report corresponding to the specified sections.
    """    
    url = f"https://api.sectors.app/v1/company/report/{ticker}/?sections={section}"

    return retrieve_from_endpoint(url)

@tool
def get_top_companies_by_tx_volume(start_date: str, end_date: str, top_n: int = 1) -> str:
    """Return a list of the most traded stocks based on transaction volume for a specified date range.
    If the question is about another country like Singapore, Thailand, India, or South Korea instead of Indonesia, inform the user: 'Sorry, I don't have the data for that country.'
    This function retrieves data to provide a list of the top traded stocks. 
    The interval can be up to 90 days. The start and end dates must be provided in the format "YYYY-MM-DD".
    This function can get data about top companies by transaction volume
    Please use function get_next_workday to check weekend or holiday date and skip it

    Args:
        start_date (str): The start date of the interval in "YYYY-MM-DD" format. Skip weekend(Saturday, Sunday) and holiday
        end_date (str): The end date of the interval in "YYYY-MM-DD" format. Skip weekend(Saturday, Sunday) and holiday
        top_n (int, optional): The number of top traded stocks to retrieve. Defaults to 1.

    Returns:
        str: A JSON string containing the list of the most traded stocks.

    """
    change = 0
    url = f"https://api.sectors.app/v1/most-traded/?start={start_date}&end={end_date}&n_stock={top_n}"
    result = retrieve_from_endpoint(url)

    if len(result)<5 and start_date!=get_today_date():
        change += 1
        start_date = pd.Timestamp(start_date)
        end_date = pd.Timestamp(end_date)
        
        # Adjust start_date if it's a weekend or holiday
        start_date = _get_next_workday_func(start_date)
        
        # Adjust end_date if it's a weekend or holiday
        end_date = _get_next_workday_func(end_date)
        url = f"https://api.sectors.app/v1/most-traded/?start={start_date}&end={end_date}&n_stock={top_n}"

    elif start_date==get_today_date():
        change += 2

    res = retrieve_from_endpoint(url)
    if start_date != end_date :
        res = get_sorted_companies_by_total_volume(json.loads(res))

    if change == 1:
        message = f"Cannot provide an answer as the specified date is a stock market holiday. Tell human you're sorry. Now the data shown for {start_date} - {end_date} is: {res}"
    elif change == 2 :
        message = f"Cannot provide an answer because the dates have not been specified and cannot provide current dates data. Tell human you're sorry"
    else:
        message = f"The data shown is: {res}"

    return message

    return memory

@tool
def get_daily_tx(ticker: str, start_date: str, end_date: str) -> str:
    """This function is designed to return the closing price, daily trading volume, and daily market capitalization.

    Args:
        ticker (str): The 4-character stock symbol representing the company. This parameter only accepts a single stock symbol.
        start_date (str): The start date of the interval in "YYYY-MM-DD" format. Skip weekend(Saturday, Sunday) and holiday
        end_date (str): The end date of the interval in "YYYY-MM-DD" format. Skip weekend(Saturday, Sunday) and holiday

    Returns:
        str: Return daily transaction data of a given ticker on a certain interval date
    """    
    url = f"https://api.sectors.app/v1/daily/{ticker}/?start={start_date}&end={end_date}"

    return retrieve_from_endpoint(url)

@tool
def get_company_performance_ipo(ticker: str) -> str:
    """Return the percentage gain since the listing date of a given ticker.
    This function retrieves data from the Sectors API to provide the performance of a specified stock
    (identified by its ticker symbol) since its initial public offering (IPO). The performance is expressed
    as a percentage gain since the listing date.

    Args:
        ticker (str): The ticker symbol of the stock for which to retrieve performance data.

    Returns:
        str: A JSON string containing the percentage gain since the stock's listing date. Sorted the days from 7, 30, 90, 365
    """    
    url = f"https://api.sectors.app/v1/listing-performance/{ticker}/"

    return retrieve_from_endpoint(url)

@tool
def duckduckgo_search(query: str) -> str:
    """Searches the web using DuckDuckGo and returns the top result.
    If the inquiry is not related to stocks, use the DuckDuckGo search feature to find the relevant information.
    """
    search = DuckDuckGoSearchRun()
    return search.run(query)

tools = [get_top_companies_by_tx_volume,
         get_next_workday,
         get_company_report,
         get_daily_tx,
         get_company_performance_ipo,
         duckduckgo_search
        ]

llm = ChatGroq(
    temperature=0.1,
    model_name="llama3-groq-70b-8192-tool-use-preview",
    groq_api_key=GROQ_API_KEY,
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            - Accepts only one stock symbol per request.
            - For multiple tickers, make separate requests for each.
            - You only have data stocks in Indonesia, if the question is about stocks in another countries like Singapore, Thailand, South Korea, tell human sorry because you don't have the data

            - Provide factual and analytical answers.
            - Include corresponding values when listing names.
            - For single-day volumes, use the same date for both start and end.
            - Present answers neatly using lists or bullet points.

            - Performance Since IPO Endpoint: Requires only the stock symbol as the parameter.

            - The key difference between get_daily_tx and get_company_report is that get_daily_tx addresses queries with a specific time or date, whereas get_company_report handles queries without a time frame.
            - Company report :
                1. Parameter required ticker and section
                2. Accepts one or more sections. Available sections include: dividend, financials, future, management, overview, ownership, peers, valuation.

            - Transaction Volume: Use the get_top_companies_by_tx_volume function whenever the query relates to the most traded stocks or transaction volume.
                1. Always consider the start and end date. 
                2. If input no spesify the dates, tell human their question must have specific dates
 
            - Daily Transation :
                1. Using tool get_daily_tx to get the closing price, daily trading volume, and daily market capitalization
                2. Please specify a ticker symbol to proceed. If none is provided, we cannot process your request.
            
            - If the inquiry is not related to stocks, use the DuckDuckGo search feature to find the relevant information.

            - If multiple questions are asked in a single input, address the first question first, and then use the response to answer the second question. Make it relevant to each other
            - Current date (Today): """ + get_today_date()
        ),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, memory=memory)

st.title("💵 IndoStock Lite")

# Initialize chat history in session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    role = message["role"]
    content = message["content"]
    st.chat_message(role).write(content)

# Input from user
if prompt := st.chat_input("Type your message here..."):
    # Append user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message
    st.chat_message("user").write(prompt)

    # Generate response
    with st.chat_message("assistant"):
        try:
            st_callback = StreamlitCallbackHandler(st.container())
            response = agent_executor.invoke({"input": prompt}, callbacks=[st_callback])
            answer = response['output']

        except Exception as e:
            answer = f"Sorry, something went wrong. I am happy to help you with any other queries you may have"

        # Append assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": answer})

        # Display assistant response
        st.write(answer)


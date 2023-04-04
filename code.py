import requests
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# World Bank API endpoint for GDP per capita data
gdp_api_url = 'http://api.worldbank.org/v2/country/{}/indicator/NY.GDP.PCAP.CD?date=2010:2020&format=json'

# World Bank API endpoint for CO2 emissions data
co2_api_url = 'http://api.worldbank.org/v2/country/{}/indicator/EN.ATM.CO2E.KT?date=2010:2020&format=json'


countries = ['USA', 'GBR', 'FRA', 'JPN', 'CAN', 'CHN', 'IND', 'PAK']

# Define indicators and years to retrieve data for Urban Population
url = 'http://api.worldbank.org/v2/country'
# Modified to include only 5 countries in Urban Population
selected_countries = ['USA', 'CAN', 'GBR', 'FRA', 'CHN']
indicator_code = 'SP.URB.TOTL'  # incdicator for urban population
start_year = 1990  # starting year for urban population
end_year = 2020  # end year for urban population
specific_country = "USA"  # only country for .describe() function
indicator_code = "SP.URB.TOTL.IN.ZS"  # indicator for USA


def fetch_data(countries, api_url):
    """
    Fetches data from World Bank API for given country codes and 
    API endpoint URL.Returns a dictionary where keys are country codes 
    and values are lists of data values for each year (2016-2020).
    """
    data = {}
    for code in countries:
        url = api_url.format(code)
        response = requests.get(url)
        if response.status_code == 200:
            # Extract data from response JSON
            values = [float(d['value']) if d['value']
                      is not None else None for d in response.json()[1]]
            data[code] = values
        else:
            print(f"Failed to fetch data for {code}")
    return data


def fetch_data_scatter_plot(
        selected_countries,
        indicator_code,
        start_year,
        end_year):
    """
    Fetches data from World Bank API for Scatter plot
    """
    url = 'http://api.worldbank.org/v2/country'
    query_url = f'{url}/{selected_countries}/indicator/{indicator_code}?format=json&date={start_year}:{end_year}'
    response = requests.get(query_url)
    data = response.json()[1]
    return pd.DataFrame(data)


def create_bar_graph(countries, data, title, y_label):
    """
    Creates a bar graph of data for given country codes and data.
    Returns the transposed dataframe.
    """
    # Create a DataFrame to hold the data
    df = pd.DataFrame(data, index=range(2010, 2021))

    # Transpose the DataFrame so that countries are columns
    df = df.transpose()

    # Set the plot size
    fig, ax = plt.subplots(figsize=(11, 6))

    # Plot the DataFrame as a bar chart
    df.plot(kind='bar', alpha=1, width=0.7, ax=ax)

    # Set chart title and axis labels
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Country Names', fontsize=16)
    ax.set_ylabel(y_label, fontsize=16)
    plt.yticks(fontsize=16)

    # Set country codes as x tick labels
    ax.set_xticklabels(countries, rotation=0, fontsize=16)

    # Set legend font size
    ax.legend(fontsize=16)

    # Show the chart
    plt.show()

    # Return the transposed dataframe
    return df.transpose()


def plot_forest_area(countries):
    """Plots forest area data on a scatter plot"""

    indicator_code = 'AG.LND.FRST.K2'
    start_year = '1990'
    end_year = '2020'
    frequency = 5

    fig, ax = plt.subplots()

    for country in countries:
        data = fetch_data_scatter_plot(
            country, indicator_code, start_year, end_year)
        data = data[data.value.notna()]
        data['year'] = pd.to_datetime(data.date).dt.year
        data = data.groupby(['year'])['value'].mean().reset_index()

        label = f'{country}'
        ax.plot(data['year'], data['value'], linestyle='--', label=label)

    ax.set_xlabel('Year')
    ax.set_ylabel('Forest Area (square kilometers)')
    ax.set_title('Forest Area for Selected Countries')
    ax.legend(bbox_to_anchor=(1, 1), loc='upper left')

    # Set x-axis ticks
    x_ticks = range(int(start_year), int(end_year) + 1, frequency)
    ax.set_xticks(x_ticks)

    plt.show()


def plot_arable_land_area(countries):
    """Plots arable land area data on a scatter plot"""

    indicator_code = 'AG.LND.ARBL.HA'
    start_year = '1990'
    end_year = '2020'
    frequency = 10

    fig, ax = plt.subplots()

    for country in countries:
        data = fetch_data_scatter_plot(
            country, indicator_code, start_year, end_year)
        data = data[data.value.notna()]
        data['year'] = pd.to_datetime(data.date).dt.year
        data = data.groupby(['year'])['value'].mean().reset_index()

        label = f'{country}'
        ax.plot(data['year'], data['value'], linestyle='--', label=label)

    ax.set_xlabel('Year')
    ax.set_ylabel('Arable Land Area (hectares)')
    ax.set_title('Arable Land Area for Selected Countries')
    ax.legend(bbox_to_anchor=(1, 1), loc='upper left')

    # Set x-axis ticks
    x_ticks = range(int(start_year), int(end_year) + 1, frequency)
    ax.set_xticks(x_ticks)

    plt.show()


def dataframe():
    """
    This Function returns two dataframes: one with years as columns 
    and one with countries as columns. cleaned transposed dataframe.
    """
    # Create a pandas dataframe from the API data
    df = pd.DataFrame(
        columns=selected_countries,
        index=range(
            start_year + 1,
            end_year))

    # Fill in the dataframe with the data for each country and year
    for code in selected_countries:
        query_url = f'{url}/{code}/indicator/{indicator_code}?format=json&date={start_year}:{end_year}'
        response = requests.get(query_url)
        data = response.json()[1]
        for i in range(len(data)):
            year = int(data[i]['date'])
            value = data[i]['value']
            if value is None:
                value = 'No data'
            else:
                value = float(value)
            df.loc[year, code] = value

    # Transpose the dataframe
    df_transposed = df.transpose()

    # Clean the transposed dataframe by resetting the index and renaming the
    # columns
    df_transposed = df_transposed.reset_index().rename(
        columns={'index': 'Country'})

    # Only keep data for every 5 years
    df_transposed_cleaned = df_transposed[df_transposed.columns[::5]]

    # Create a pandas dataframe from the API data
    df = pd.DataFrame(
        columns=range(
            start_year + 5,
            end_year),
        index=selected_countries)

    # Fill in the dataframe with the data for each country and year
    for code in selected_countries:
        query_url = f'{url}/{code}/indicator/{indicator_code}?format=json&date={start_year}:{end_year}'
        response = requests.get(query_url)
        data = response.json()[1]
        for i in range(len(data)):
            year = int(data[i]['date'])
            value = data[i]['value']
            if value is None:
                value = 'No data'
            else:
                value = float(value)
            df.loc[code, year] = value

    # Clean the dataframe by dropping columns that don't fall on a 5-year
    # interval
    df_cleaned = df[df.columns[::5]]

    # Transpose the dataframe
    df_transposed = df_cleaned.transpose()

    return df_transposed_cleaned, df_transposed


def describe_method(specific_country, indicator_code):
    """
    This function explore the data with .describe() method
    and produce the statistical properties of a few indicators
    """

    # Set up the URL for the API request
    url = "http://api.worldbank.org/v2/country/{}/indicator/{}?format=json".format( 
        specific_country, indicator_code)

    # Send a GET request to the API endpoint
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Extract the data from the response JSON object
        data = response.json()[1]

        # Create a DataFrame from the data
        df = pd.DataFrame(data)

        # Rename the columns to something more readable
        df = df.rename(columns={"date": "Year", "value": "Indicator Value"})

        # Set the index to be the year
        df = df.set_index("Year")

        # Generate descriptive statistics for the indicator data
        stats = df["Indicator Value"].describe()

        return stats
    else:
        print("Error: Could not retrieve data from World Bank API.")
        return None


if __name__ == '__main__':
    # Randomly selected 5 country codes

    # Fetch GDP per capita data for each country
    gdp_data = fetch_data(countries, gdp_api_url)

    # Create bar graph for GDP per capita data, get the transposed dataframe
    gdp_df = create_bar_graph(
        countries,
        gdp_data,
        'GDP per capita (2016-2020)',
        'GDP per capita (current US$)')

    # Fetch CO2 emissions data for each country
    co2_data = fetch_data(countries, co2_api_url)

    # Create bar graph for CO2 emissions data and get the transposed dataframe
    co2_df = create_bar_graph(
        countries,
        co2_data,
        'CO2 emissions (2016-2020)',
        'CO2 emissions (kt)')

    plot_forest_area(countries)
    plot_arable_land_area(countries)

    df_transposed_cleaned, df_transposed = dataframe()
    print("\nFirst dataframe for Urban Population:\n")
    print(dataframe())

    stats = describe_method(specific_country, indicator_code)
    print("\nDescribe Method For USA\n")
    print(stats)

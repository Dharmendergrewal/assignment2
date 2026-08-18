import requests
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import skew

# Set modern Seaborn theme globally
sns.set_theme(style="whitegrid", context="talk", palette="deep")

# World Bank API endpoint for GDP per capita data
gdp_api_url = 'http://api.worldbank.org/v2/country/{}/indicator/NY.GDP.PCAP.CD?date=2010:2020&format=json'

# NOTE: Replaced CO2 emissions data with Population Density (EN.POP.DNST) 
# because the original CO2 indicator was deleted by the World Bank.
pop_density_api_url = 'http://api.worldbank.org/v2/country/{}/indicator/EN.POP.DNST?date=2010:2020&format=json'

countries = ['USA', 'GBR', 'FRA', 'JPN', 'CAN', 'CHN', 'IND', 'PAK']

# Define indicators and years to retrieve data for Urban Population
url = 'http://api.worldbank.org/v2/country'
selected_countries = ['USA', 'CAN', 'GBR', 'FRA', 'CHN']
indicator_code = 'SP.URB.TOTL'  # indicator for urban population
start_year = 1990  # starting year for urban population
end_year = 2020  # end year for urban population
specific_country = "CHN"  # only country for .describe() function
indicator_code_usa = "SP.URB.TOTL.IN.ZS"  # indicator for USA


def fetch_data(countries, api_url):
    """
    Fetches data from World Bank API for given country codes and
    API endpoint URL. Returns a dictionary where keys are country codes
    and values are lists of data values for each year.
    """
    data = {}
    for code in countries:
        query_url = api_url.format(code)
        try:
            response = requests.get(query_url)
            if response.status_code == 200:
                resp_json = response.json()
                if len(resp_json) > 1 and isinstance(resp_json[1], list):
                    values = [float(d['value']) if d.get('value') is not None else None for d in resp_json[1]]
                    data[code] = values
                else:
                    print(f"Warning: Failed to fetch valid data for {code}. API returned: {resp_json}")
                    data[code] = []
            else:
                print(f"Failed to fetch data for {code}. Status Code: {response.status_code}")
        except Exception as e:
            print(f"Error fetching data for {code}: {e}")
            data[code] = []
    return data


def fetch_data_scatter_plot(selected_countries, indicator_code, start_year, end_year):
    """
    Fetches data from World Bank API for Scatter plot
    """
    query_url = f'{url}/{selected_countries}/indicator/{indicator_code}?format=json&date={start_year}:{end_year}'
    try:
        response = requests.get(query_url)
        if response.status_code == 200:
            resp_json = response.json()
            if len(resp_json) > 1 and isinstance(resp_json[1], list):
                return pd.DataFrame(resp_json[1])
    except Exception as e:
        print(f"Error fetching scatter data for {selected_countries}: {e}")
    return pd.DataFrame()


def create_bar_graph(countries, data, title, y_label):
    """
    Creates a bar graph of data for given country codes and data.
    """
    # Create a DataFrame to hold the data
    df = pd.DataFrame(data, index=range(2010, 2021)).transpose()
    
    # Drop empty rows to avoid plotting errors
    df = df.dropna(how='all')

    if df.empty:
        print(f"No data available to plot for: {title}")
        return pd.DataFrame()

    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Modern bar plot using seaborn palette
    df.plot(kind='bar', width=0.8, ax=ax, colormap="viridis", edgecolor='none')

    ax.set_title(title, fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('Country', fontsize=14, labelpad=10)
    ax.set_ylabel(y_label, fontsize=14, labelpad=10)
    
    ax.set_xticklabels(df.index, rotation=0, fontsize=12)
    plt.yticks(fontsize=12)

    # Clean legend and put it outside
    ax.legend(title="Year", bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=12, frameon=False)
    
    sns.despine(left=True, bottom=False)
    plt.tight_layout()
    plt.show()

    return df.transpose()


def scatter_forest(countries):
    """Plots forest area data on a scatter plot"""
    indicator_code = 'AG.LND.FRST.K2'
    start_year = '1990'
    end_year = '2020'
    frequency = 5

    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = sns.color_palette("husl", len(countries))
    plotted = False

    for idx, country in enumerate(countries):
        data = fetch_data_scatter_plot(country, indicator_code, start_year, end_year)
        if not data.empty and 'value' in data.columns and data['value'].notna().any():
            data = data[data.value.notna()].copy()
            data['year'] = pd.to_datetime(data.date).dt.year
            data = data.groupby(['year'])['value'].mean().reset_index()

            sns.lineplot(
                x='year', y='value', data=data, 
                marker='o', markersize=8, linewidth=2.5, 
                label=f'{country}', color=colors[idx], ax=ax
            )
            plotted = True

    if not plotted:
        print("No forest data plotted.")
        plt.close()
        return

    ax.set_xlabel('Year', fontsize=14, labelpad=10)
    ax.set_ylabel('Forest Area (sq km)', fontsize=14, labelpad=10)
    ax.set_title('Forest Area Dynamics', fontsize=18, fontweight='bold', pad=20)
    
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False)

    x_ticks = range(int(start_year), int(end_year) + 1, frequency)
    ax.set_xticks(x_ticks)
    
    sns.despine(left=True, bottom=False)
    plt.tight_layout()
    plt.show()


def scatter_arable(countries):
    """Plots arable land area data on a scatter plot"""
    indicator_code = 'AG.LND.ARBL.HA'
    start_year = '1990'
    end_year = '2020'
    frequency = 5

    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = sns.color_palette("Set2", len(countries))
    plotted = False

    for idx, country in enumerate(countries):
        data = fetch_data_scatter_plot(country, indicator_code, start_year, end_year)
        if not data.empty and 'value' in data.columns and data['value'].notna().any():
            data = data[data.value.notna()].copy()
            data['year'] = pd.to_datetime(data.date).dt.year
            data = data.groupby(['year'])['value'].mean().reset_index()

            sns.lineplot(
                x='year', y='value', data=data, 
                marker='D', markersize=8, linewidth=2.5, 
                label=f'{country}', color=colors[idx], ax=ax
            )
            plotted = True

    if not plotted:
        print("No arable land data plotted.")
        plt.close()
        return

    ax.set_xlabel('Year', fontsize=14, labelpad=10)
    ax.set_ylabel('Arable Land Area (hectares)', fontsize=14, labelpad=10)
    ax.set_title('Arable Land Area Over Time', fontsize=18, fontweight='bold', pad=20)
    
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False)

    x_ticks = range(int(start_year), int(end_year) + 1, frequency)
    ax.set_xticks(x_ticks)
    
    sns.despine(left=True, bottom=False)
    plt.tight_layout()
    plt.show()


def country_correlation_heatmap(countries, start_date, end_date):
    """
    Creates a correlation matrix of countries based on their GDP per capita and CO2 emissions.
    Matches the exact layout of the user's requested heatmap.
    """
    gdp_ind = 'NY.GDP.PCAP.CD'
    # Note: EN.ATM.CO2E.KT is dead, using CC.CO2.EMSE.IL as fallback
    co2_ind = 'CC.CO2.EMSE.IL' 
    
    data = {}
    for country in countries:
        country_data = []
        for ind in [gdp_ind, co2_ind]:
            query = f'http://api.worldbank.org/v2/country/{country}/indicator/{ind}?format=json&date={start_date}:{end_date}'
            try:
                response = requests.get(query)
                if response.status_code == 200:
                    resp_json = response.json()
                    if len(resp_json) > 1 and isinstance(resp_json[1], list):
                        # Sort by date to maintain chronological order
                        vals = sorted(resp_json[1], key=lambda x: x['date'])
                        country_data.extend([float(d['value']) if d.get('value') is not None else 0 for d in vals])
            except Exception as e:
                print(f"Error fetching correlation data for {country}: {e}")
                
        if country_data:
            data[country] = country_data

    if not data:
        print("No valid data for heatmap.")
        return

    df = pd.DataFrame(data)
    corr_matrix = df.corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Highly understandable heatmap with annotations and clear scaling
    sns.heatmap(
        corr_matrix, 
        annot=True,           # Show numbers in boxes
        fmt=".2f",            # Limit to 2 decimal places
        cmap='coolwarm', 
        vmin=-1, vmax=1,      # Lock color scale to -1 and 1
        center=0,
        square=True, 
        linewidths=0.5,       # Add borders between boxes for clarity
        cbar_kws={"shrink": .8, "label": "Correlation Coefficient"},
        ax=ax
    )
    
    plt.title(f'Correlation between GDP & CO2 for Selected Countries ({start_date}-{end_date})', 
              pad=20, fontsize=14, fontweight='bold')
    
    # Clean up the axes for maximum readability
    plt.xticks(rotation=0, fontsize=11)
    plt.yticks(rotation=0, fontsize=11)
    plt.xlabel('Countries', fontsize=12, labelpad=10)
    plt.ylabel('Countries', fontsize=12, labelpad=10)
    
    plt.tight_layout()
    plt.show()


def dataframe():
    """
    Returns two dataframes: cleaned transposed dataframe and transposed dataframe.
    """
    df = pd.DataFrame(columns=selected_countries, index=range(start_year + 1, end_year))

    for code in selected_countries:
        query_url = f'{url}/{code}/indicator/{indicator_code}?format=json&date={start_year}:{end_year}'
        try:
            response = requests.get(query_url)
            if response.status_code == 200:
                resp_json = response.json()
                if len(resp_json) > 1 and isinstance(resp_json[1], list):
                    data = resp_json[1]
                    for i in range(len(data)):
                        year = int(data[i]['date'])
                        value = data[i]['value']
                        df.loc[year, code] = float(value) if value is not None else np.nan
        except Exception as e:
            print(f"Error processing dataframe for {code}: {e}")

    df_transposed = df.transpose()
    df_transposed = df_transposed.reset_index().rename(columns={'index': 'Country'})
    
    # Cleaned dataframe every 5 years
    df_transposed_cleaned = df_transposed[df_transposed.columns[::5]]

    return df_transposed_cleaned, df_transposed


def describe_method(specific_country, ind_code):
    """
    Explores the data with .describe() method and statistical properties
    """
    query_url = f"http://api.worldbank.org/v2/country/{specific_country}/indicator/{ind_code}?format=json"
    try:
        response = requests.get(query_url)
        if response.status_code == 200:
            resp_json = response.json()
            if len(resp_json) > 1 and isinstance(resp_json[1], list):
                data = resp_json[1]
                df = pd.DataFrame(data)
                df = df.rename(columns={"date": "Year", "value": "Indicator Value"})
                df = df.set_index("Year")
                
                # Convert to numeric
                df["Indicator Value"] = pd.to_numeric(df["Indicator Value"], errors='coerce')
                df = df.dropna(subset=["Indicator Value"])

                if df.empty:
                    return None

                stats = df["Indicator Value"].describe()
                skewness = skew(df["Indicator Value"])
                stats["Skewness"] = skewness
                
                return stats
    except Exception as e:
        print(f"Error fetching describe data: {e}")
    
    print("Could not retrieve valid data from World Bank API.")
    return None


if __name__ == '__main__':
    # Fetch GDP per capita data for each country
    gdp_data = fetch_data(countries, gdp_api_url)

    # Create bar graph for GDP per capita data
    gdp_df = create_bar_graph(
        countries,
        gdp_data,
        'GDP per capita (2010-2020)',
        'GDP per capita (current US$)')

    # Fetch Population Density data instead of CO2 emissions (since CO2 is broken in API)
    pop_density_data = fetch_data(countries, pop_density_api_url)

    # Create bar graph for Population density
    pop_density_df = create_bar_graph(
        countries,
        pop_density_data,
        'Population Density (2010-2020)',
        'People per sq. km of land area')

    # create a correlation heatmap of countries for GDP and CO2 (2016-2020)
    country_correlation_heatmap(countries, '2016', '2020')

    scatter_forest(countries)
    scatter_arable(countries)

    df_transposed_cleaned, df_transposed = dataframe()
    print("\nFirst dataframe for Urban Population:\n")
    print(df_transposed_cleaned)

    # Use the USA indicator_code_usa for describe
    stats = describe_method(specific_country, indicator_code_usa)
    if stats is not None:
        print("\nStats for China\n")
        print(stats)

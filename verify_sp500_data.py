import pandas as pd

def verify_sp500_sectors():
    """
    Reads S&P 500 company list from Wikipedia and prints verification data.
    """
    try:
        # Wikipedia URL for the list of S&P 500 companies
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        
        # pandas.read_html reads all tables from the URL into a list of DataFrames
        # Add a User-Agent header to avoid HTTP Error 403: Forbidden
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        tables = pd.read_html(url, storage_options={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'})
        
        # The first table on the page is the one we need
        sp500_df = tables[0]
        
        print("--- S&P 500 Data Verification ---")
        
        # 1. Show the first 5 rows to verify the structure
        print("\n[1] First 5 rows of the table:")
        print(sp500_df.head())
        
        # 2. Show the total number of companies found
        print(f"\n[2] Total companies found: {sp500_df.shape[0]}")
        
        # 3. Show all unique sectors found in the data
        print("\n[3] Unique sectors found:")
        print(sp500_df['GICS Sector'].unique())
        
        # 4. Count companies in the 'Information Technology' sector
        it_sector_count = sp500_df[sp500_df['GICS Sector'] == 'Information Technology'].shape[0]
        print(f"\n[4] Number of companies in 'Information Technology' sector: {it_sector_count}")
        
        print("\n--- Verification Complete ---")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    verify_sp500_sectors()

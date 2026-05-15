from selenium import webdriver
from bs4 import BeautifulSoup
import pandas as pd
import time

url = "https://www.shl.com/products/product-catalog/"

driver = webdriver.Chrome()
driver.get(url)

time.sleep(5)  # wait for JS load

soup = BeautifulSoup(driver.page_source, "html.parser")

driver.quit()

data = []

rows = soup.find_all("tr")

for row in rows:
    cols = row.find_all("td")

    if len(cols) >= 2:
        a = cols[0].find("a")

        if a:
            name = a.text.strip()
            link = "https://www.shl.com" + a["href"]
            test_type = cols[-1].text.strip()

            data.append({
                "name": name,
                "url": link,
                "test_type": test_type
            })

df = pd.DataFrame(data)
df.to_csv("catalog.csv", index=False)

print(df.head())
print("Saved catalog.csv")
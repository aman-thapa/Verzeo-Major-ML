from bs4 import BeautifulSoup
import cvlib as cv
import cv2, sys, time, os, requests
from skimage import io
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.firefox.options import Options

import DataCleaning


options = Options()
options.set_preference('permissions.default.image', 2)
options.set_preference('dom.ipc.plugins.enabled.libflashplayer.so', False)

Google_Image = \
    'https://www.google.com/search?tbm=isch&num=10000&'


u_agnt = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.83 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
    'Accept-Encoding': 'none',
    'Accept-Language': 'en-US,en;q=0.8',
    'Connection': 'keep-alive',
}
Image_Folder = 'non-indian-dataset'

def main():
    data = ["Australian", "Korean", "Chinese", "Japanese", "Amarican", "Canadian", "Germany", "Russian", "Thailand", "African"]
    if not os.path.exists(Image_Folder):
        os.mkdir(Image_Folder)
    for i in data:
        download_images(i)

def download_images(data):

    print(f'Searching {data} Images....')
    driver = webdriver.Firefox(options=options ,executable_path='./geckodriver')
    search_url = Google_Image + 'q=' + data + "Human faces"#'q=' because its a query
    driver.get(search_url)
    pause = 2
    time.sleep(pause)
    while True:
        lastHeight = driver.execute_script("return document.body.scrollHeight")
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(pause)
        newHeight = driver.execute_script("return document.body.scrollHeight")
        if newHeight == lastHeight:
            try:
                b_soup = BeautifulSoup(driver.page_source, 'html.parser')
                more = driver.find_elements_by_xpath('/html/body/div[2]/c-wiz/div[3]/div[1]/div/div/div/div/div[4]/div[2]/input')[0]
                more.click()
                time.sleep(5)
            except Exception as e:
                print(e)
                break
        #print(lastHeight)
    b_soup = BeautifulSoup(driver.page_source, 'html.parser')

    results = b_soup.findAll('img', {'class': 'rg_i Q4LuWd'})
    driver.close()
    imagelinks= []
    print("Got : ",len(results), "images")
    for res in results:
        try:
            link = res['data-src']
            try:
                image = io.imread(link)
                if image.shape[-1] == 4:
                    image = cv2.cvtColor(image,cv2.COLOR_RGBA2BGR)
                if image.shape[-1] == 1:
                    continue
                faces, confidence = cv.detect_face(image)
                if len(faces) != 1 : #if no Face or more than one face detected :: Skip Link
                    continue
            except:
                continue
            imagelinks.append(link)
        except KeyError:
            continue

    print(f'Found {len(imagelinks)} images')
    print('Start downloading...')

    for i, imagelink in enumerate(imagelinks):
        response = requests.get(imagelink)

        imagename = Image_Folder + '/' + data + '-' + str(i+1) + '.jpg'
        with open(imagename, 'wb') as file:
            file.write(response.content)

    print('Download Completed!')

if __name__ == '__main__':
    main()
    Dupilcates.function(Image_Folder)

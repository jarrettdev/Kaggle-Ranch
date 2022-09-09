// it augments the installed puppeteer with plugin functionality
const puppeteer = require('puppeteer-extra')

const AdblockerPlugin = require('puppeteer-extra-plugin-adblocker')
const StealthPlugin = require('puppeteer-extra-plugin-stealth');
const fs = require('fs');
puppeteer.use(AdblockerPlugin());

(async () => {
    puppeteer.use(
        require('puppeteer-extra-plugin-stealth/evasions/navigator.webdriver')()
    );
    puppeteer.use(
        require('puppeteer-extra-plugin-stealth/evasions/sourceurl')()
    );
    try {
        const customArgs = [
            `--start-maximized`,
        ];
        //
        try {
            var searchQuery = myArgs[0];
            const userId = myArgs[1];
        } catch {
            console.log('No search query provided');
        }
        const browser = await puppeteer.launch({
            defaultViewport: null,
            ignoreHTTPSErrors: true,
            executablePath: process.env.chrome,
            headless: false,
            ignoreDefaultArgs: ["--disable-extensions", "--enable-automation"],
            args: customArgs,
        });
        const testUrl = 'https://abrahamjuliot.github.io/creepjs/';
        const page = await browser.newPage();
        await page.waitForTimeout(2000)
        try {
            for (let i = 1; i < 30; i++) {
                url = `https://www.kaggle.com/datasets?topic=trendingDataset&page=${i}`;
                await page.goto(url);
                await page.waitForSelector('li[role="listitem"]');
                const cardContainer = await page.$(
                    'ul[class="km-list km-list--three-line"]'
                )
                console.log(cardContainer.length);
                //each card is an li element inside of cardContainer
                await page.waitForTimeout(2000);
                cards = await page.$$('li');
                console.log('Kaggle cards found\n==================');
                console.log(cards.length);
                for (let i = 0; i < cards.length; i++) {
                    const card = cards[i];
                    try {
                        const cardTitle = await card.$eval('div a', (el) => el.innerText);
                        const cardLink = await card.$eval('div a', (el) => el.href);
                        console.log(cardTitle);
                        console.log(cardLink);
                        if (cardLink) {
                            //append to file
                            fs.appendFile('kaggle.txt', cardLink + '\n', function (err) { });
                        }
                        console.log('==================');

                    }
                    catch (err) {
                    }

                }
            }

        } catch (err) {
            console.error(err);
        }

        console.log('Done');
    }
    catch (err) {
        console.log(err);
    }
})();

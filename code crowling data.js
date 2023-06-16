
async function openUrl(url) {
  try {
    const browser = await puppeteerExtra.launch({
      headless: true,
      args: [
        "--no-sandbox",
        "--disable-setuid-sandbox",
        "--disable-infobars",
        "--disable-dev-shm-usage",
        "--disable-browser-side-navigation",
        "--disable-features=site-per-process",
        "--disable-web-security",
      ],
    });
    const page = await browser.newPage();

    const userAgent = randomUseragent.getRandom();
    await page.setUserAgent(userAgent);

    const { width, height } = await page.evaluate(() => ({
      width: window.screen.width,
      height: window.screen.height,
    }));
    await page.setViewport({ width, height });

    await page.goto(url);

    return page;
  } catch (error) {
    console.error(error);
    throw error;
  }
}

async function extractUrlsFromPages(
  initialUrl,
  totalPages,
  urlFormat,
  page_now
) {
  const urls = [];

  const browser = await puppeteer.launch();
  const page = await browser.newPage();

  for (let currentPage = 1; currentPage <= totalPages; currentPage++) {
    const url = urlFormat.replace("{page}", currentPage);

    await page.goto(url);
    console.log(`Extracting links from: ${url}`);

    const links = await page.$$eval(".item-container a.item-img", (elements) =>
      elements.map((element) => element.href.trim())
    );

    urls.push(...links);
    console.log(`Found ${links.length} links on page ${currentPage}`);
    await wait(3000);
  }

  await browser.close();

  return urls;
}

async function run() {
  const pagenow = 1; // Define pagenow before using it
  const initialUrl1 =
    "https://www.newegg.com/Business-Laptops/SubCategory/ID-3413/Page-{page}?Tid=167751";
  const totalPages1 = 35;
  const urlFormat1 = initialUrl1;
  const urls1 = await extractUrlsFromPages(
    initialUrl1,
    totalPages1,
    urlFormat1,
    pagenow
  );

  const pagenow2 = 1; // Define pagenow2 before using it
  const initialUrl2 =
    "https://www.newegg.com/Laptops-Notebooks/SubCategory/ID-32/Page-{page}?Tid=6770";
  const totalPages2 = 100;
  const urlFormat2 = initialUrl2;
  const urls2 = await extractUrlsFromPages(
    initialUrl2,
    totalPages2,
    urlFormat2,
    pagenow2
  );

  const urls = [...urls1, ...urls2];

  return urls; // Return the obtained URLs
}
async function scrapePrices() {
  try {
    const urls = await run();
    console.log(urls);
    const laptopsData = [];

    const browser = await puppeteer.launch();
    const page = await browser.newPage();

    for (const [index, url] of urls.entries()) {
      await page.goto(url, { timeout: 60000 });
      try {
        await page.waitForSelector(".product-view-brand.has-brand-store .logo");

        const condition = await page.evaluate(() => {
          const conditionElement = document.querySelector(
            ".product-offer .link-more"
          );
          if (conditionElement) {
            const conditionText = conditionElement.innerText;
            if (conditionText.toLowerCase().includes("new")) {
              return 1;
            }
          }
          return 0;
        });

        console.log(`Condition: ${condition}`);

        const brand = await page.evaluate(() => {
          const brandElement = document.querySelector(
            ".product-view-brand.has-brand-store .logo"
          );
          if (brandElement) {
            return brandElement.getAttribute("alt");
          }
          return null;
        });

        const price = await page.evaluate(() => {
          const priceElement = document.querySelector(
            ".product-price .price-current"
          );
          if (priceElement) {
            const priceText = priceElement.innerText;
            return priceText;
          }
          return null;
        });
        console.log(`Price: ${price}`);

        const isButtonClicked = await page.evaluate(() => {
          const button = document.querySelector(".tab-nav.active");
          if (button) {
            button.click();
            return true;
          }
          return false;
        });

        if (isButtonClicked) {
          console.log("The button was clicked successfully.", url);
          await wait(5000);
        } else {
          console.log(
            "Failed to click the button or expected changes did not occur for URL:",
            url
          );
        }

        const Memory = await searchTableValue(page, "Memory", "Memory");
        console.log(`Value: ${Memory}`);

        const cpu_speed = await searchTableValue(page, "CPU Speed", "CPU");
        console.log(`Value: ${cpu_speed}`);

        const Number_of_Cores = await searchTableValue(
          page,
          " Number of Cores",
          "CPU"
        );
        const extractedNumber =
          Number_of_Cores && Number_of_Cores.length > 0
            ? Number_of_Cores[0].match(/\d+/)?.[0]
            : null;
        console.log(`Value3: ${extractedNumber}`);

        const sizescreen = await searchTableValue(
          page,
          "Screen Size",
          "Display"
        );
        console.log(`Value: ${sizescreen}`);

        const year = await searchTableValue(
          page,
          "Date First Available",
          "Additional Information"
        );
        const cleanedYear = year ? year.match(/\d{4}$/)?.[0] : null;
        console.log(`Value: ${cleanedYear}`);

        const Resolution = await searchTableValue(
          page,
          "Resolution",
          "Display"
        );
        console.log(`Value: ${Resolution}`);

        const ssd = await searchTableValue(page, "SSD", "Storage");
        console.log(`Value: ${ssd}`);

        const touchscreen = await searchTableValue(
          page,
          "Touchscreen ",
          "Display"
        );
        const touchscreen_value = touchscreen === "No" ? 0 : 1;
        console.log(`Value: ${touchscreen_value}`);

        const Webcam = await searchTableValue(
          page,
          "Webcam",
          "Supplemental Drive"
        );
        const Webcam_value = Webcam === "No" ? 0 : 1;
        console.log(`Value1: ${Webcam_value}`);

        const opticaldrive = await searchTableValue(
          page,
          "Optical Drive Type",
          "Optical Drive"
        );
        const opticaldrive_value = opticaldrive === "No" ? 0 : 1;
        console.log(`Value2: ${opticaldrive_value}`);

        const Type = await searchTableValue(page, "Type", "General");
        console.log(`Value: ${Type}`);

        const Weight = await searchTableValue(
          page,
          "Weight",
          "Dimensions & Weight"
        );
        console.log(`Value: ${Weight}`);

        const bluetooth = await searchTableValue(
          page,
          "Bluetooth",
          "Communications"
        );

        const USB = await searchTableValue(page, "USB", "Ports");

        const connectionsRegex = /(\d+)\s*x\s*USB/g;
        const typeRegex = /USB/g;

        let connectionsMatch = USB.match(connectionsRegex);
        let typeMatch = USB.match(typeRegex);

        // Ensure that connectionsMatch and typeMatch are arrays
        connectionsMatch = Array.isArray(connectionsMatch)
          ? connectionsMatch
          : [];
        typeMatch = Array.isArray(typeMatch) ? typeMatch : [];

        const numDifferentTypes = typeMatch.length;
        let totalConnections = 0;

        for (const match of connectionsMatch) {
          const connectionCount = parseInt(match.match(/\d+/)[0]);
          totalConnections += connectionCount;
        }

        console.log(`USB connections of different types: ${numDifferentTypes}`);
        console.log(`Total USB connections: ${totalConnections}`);

        const laptopData = {
          Condition: condition,
          Brand: brand,
          Price: price,
          ProcessorSpeed: cpu_speed,
          RAMSize: Memory,
          NumberOfCores: extractedNumber,
          ScreenSize: sizescreen,
          ReleaseYear: cleanedYear,
          Resolution: Resolution,
          SSD: ssd,
          Touchscreen: touchscreen_value,
          Webcam: Webcam_value,
          OpticalDrive: opticaldrive_value,
          Type: Type,
          Weight: Weight,
          BluetoothVersion: bluetooth,
          NumDifferentUSBTypes: numDifferentTypes,
          TotalUSBConnections: totalConnections,
        };

        laptopsData.push(laptopData);
        await wait(5000); // Add delay after button click
      } catch (error) {
        if (error.name === "TimeoutError") {
          console.error(`Navigation timeout occurred on URL: ${url}`);
          continue; // Skip to the next iteration
        }

        console.error(`Error occurred while processing URL: ${url}`);
        console.error(error);
        continue; // Skip to the next iteration
      }
    }

    await page.close();
    await browser.close();

    console.log(laptopsData);

    const jsonData = JSON.stringify(laptopsData);
    fs.writeFileSync("laptops_data.json", jsonData);
    console.log("Data saved to laptops_data.json");
  } catch (error) {
    console.error("An error occurred:", error);
  }
}

scrapePrices();

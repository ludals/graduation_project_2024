import { useState, useEffect } from "react";
import styled from "styled-components";
import StockMarketMap from "./StockMarketMap";
import { Companies } from "./companies";

export default function StartPage() {
  const [marketData, setMarketData] = useState([]);

  useEffect(() => {
    loadAllCSVData();
  }, []);

  const loadAllCSVData = async () => {
    const loadedData = [];

    for (const [companyName, companyValue] of Object.entries(Companies)) {
      try {
        const response = await fetch(`/data/${companyValue}.csv`);
        const data = await response.text();
        const parsedData = parseCSVData(data);

        if (parsedData.length >= 2) {
          const latestData = parsedData[parsedData.length - 1];
          const previousData = parsedData[parsedData.length - 2];

          const change =
            ((latestData.close - previousData.close) / previousData.close) *
            100;

          loadedData.push({
            name: companyName,
            change,
          });
        } else {
          console.warn(`Not enough data for ${companyName}`);
          loadedData.push({
            name: companyName,
            change: 0,
          });
        }
      } catch (error) {
        console.error(`Error loading data for ${companyName}:`, error);
        loadedData.push({
          name: companyName,
          change: 0,
        });
      }
    }

    setMarketData(loadedData);
  };

  const parseCSVData = (data) => {
    const rows = data.trim().split("\n");
    return rows.slice(1).map((row) => {
      const values = row.split(",");
      return {
        date: values[0],
        open: parseInt(values[1], 10),
        high: parseInt(values[2], 10),
        low: parseInt(values[3], 10),
        close: parseInt(values[4], 10),
        volume: parseInt(values[5], 10),
      };
    });
  };

  return (
    <Container>
      <MarketBox>
        <StockMarketMap stockData={marketData} isPrediction={false} />
        <StockMarketMap stockData={marketData} isPrediction={true} />
      </MarketBox>
    </Container>
  );
}

const Container = styled.div`
  display: flex;
  justify-content: center;
  align-items: center;
  width: 100%;
  height: 100%;
`;

const MarketBox = styled.div`
  display: flex;
  gap: 30px;
  flex-direction: row;
  width: 100%;
  height: 100%;
`;

import { useState, useEffect } from "react";
import styled from "styled-components";
import StockMarketMap from "./StockMarketMap";
import { Companies } from "./companies";

export default function StartPage() {
  const [marketData, setMarketData] = useState([]);
  const [predictionData, setPredictionData] = useState([]);

  useEffect(() => {
    loadAllCSVData();
  }, []);

  // marketData가 업데이트될 때 loadPredictionData 호출
  useEffect(() => {
    if (marketData.length > 0) {
      loadPredictionData(); // 예측 데이터를 로드
    }
  }, [marketData]);

  // 현재 시장 데이터를 로드
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
            close: latestData.close, // 현재 종가
            change,
          });
        } else {
          console.warn(`Not enough data for ${companyName}`);
          loadedData.push({
            name: companyName,
            close: 0,
            change: 0,
          });
        }
      } catch (error) {
        console.error(`Error loading data for ${companyName}:`, error);
        loadedData.push({
          name: companyName,
          close: 0,
          change: 0,
        });
      }
    }

    setMarketData(loadedData);
  };

  // 예측 데이터를 로드
  const loadPredictionData = async () => {
    try {
      const response = await fetch(
        "http://localhost:5000/api/predict-next-day"
      );
      const predictionResults = await response.json();

      const predictionChanges = predictionResults.map((prediction) => {
        const { ticker, predictedNextDayPrice } = prediction;
        const currentData = marketData.find((data) => data.name === ticker);

        // 실제 종가와 예측 종가의 차이 계산
        if (currentData && currentData.close) {
          const predictedChange =
            ((predictedNextDayPrice - currentData.close) / currentData.close) *
            100;
          return {
            name: ticker,
            close: predictedNextDayPrice, // 예측 종가
            change: predictedChange,
          };
        } else {
          console.warn(`No close data available for ${ticker}`);
          return {
            name: ticker,
            close: predictedNextDayPrice, // 예측 종가 사용
            change: 0,
          };
        }
      });

      setPredictionData(predictionChanges);
      console.log("Prediction data loaded:", predictionChanges);
    } catch (error) {
      console.error("Error loading prediction data:", error);
    }
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
        <StockMarketMap stockData={predictionData} isPrediction={true} />
      </MarketBox>
    </Container>
  );
}

const Container = styled.div`
  display: flex;
  justify-content: center;
  align-items: center;
  height: 98vh;
`;

const MarketBox = styled.div`
  display: flex;
  gap: 30px;
  flex-direction: row;
  width: 100%;
  height: 100%;

  overflow: hidden;
`;

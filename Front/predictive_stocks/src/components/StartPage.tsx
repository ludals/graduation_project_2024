import { useState, useEffect } from "react";
import styled from "styled-components";
import StockMarketMap from "./StockMarketMap";
import { Companies } from "./companies";

export default function StartPage() {
  const [marketData, setMarketData] = useState([]);
  const [predictionData, setPredictionData] = useState([]);
  const [loadingMarketData, setLoadingMarketData] = useState(true);
  const [loadingPredictionData, setLoadingPredictionData] = useState(true);

  // 초기 로드
  useEffect(() => {
    loadAllCSVData();
  }, []);

  // marketData가 로드된 후에만 loadPredictionData 호출
  useEffect(() => {
    if (!loadingMarketData) {
      loadPredictionData();
    }
  }, [loadingMarketData]);

  // 현재 시장 데이터를 로드
  const loadAllCSVData = async () => {
    const loadedData = [];
    setLoadingMarketData(true);
    for (const [companyName, companyValue] of Object.entries(Companies)) {
      try {
        const response = await fetch(`/data/${companyValue}.csv`);
        const data = await response.text();
        const parsedData = parseCSVData(data);

        if (parsedData.length >= 2) {
          // 가장 마지막 값이 최신 데이터, 그 전 값이 이전 데이터
          const latestData = parsedData[0];
          const previousData = parsedData[1];

          const change =
            ((latestData.close - previousData.close) / previousData.close) *
            100;

          loadedData.push({
            name: companyName,
            close: latestData.close, // 최신 종가
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

    console.log("Final loaded data:", loadedData); // 최종 결과 확인

    setMarketData(loadedData);
    setLoadingMarketData(false);
  };

  // 예측 데이터를 로드
  const loadPredictionData = async () => {
    setLoadingPredictionData(true);
    const response = await fetch("http://localhost:5000/api/predict-next-day");
    const predictionResults = await response.json();
    console.log("Prediction data:", predictionResults);

    const predictionChanges = predictionResults.map((prediction) => {
      const ticker = prediction.ticker;
      const predictedNextDayPrice = prediction.predictedNextDayPrice;

      // marketData에서 ticker와 일치하는 데이터를 찾기
      const currentData = marketData.find((data) => data.name === ticker);
      console.log("Current data for", ticker, ":", currentData);
      if (!currentData) {
        console.warn(`Ticker ${ticker} not found in marketData`);
      }

      if (currentData && currentData.close) {
        const predictedChange =
          ((predictedNextDayPrice - currentData.close) / currentData.close) *
          100;
        return {
          name: ticker,
          close: predictedNextDayPrice,
          change: predictedChange,
        };
      } else {
        return {
          name: ticker,
          close: predictedNextDayPrice,
          change: 0,
        };
      }
    });
    console.log("Final prediction data:", predictionChanges);
    setPredictionData(predictionChanges);
    setLoadingPredictionData(false);
  };
  const parseCSVData = (data) => {
    const rows = data.trim().split("\n");
    const parsedData = rows.slice(1).map((row) => {
      const values = row.split(",");
      const parsedRow = {
        date: values[0],
        open: parseFloat(values[1]),
        high: parseFloat(values[2]),
        low: parseFloat(values[3]),
        close: parseFloat(values[4]), // 여기서 close 값을 정확히 변환하는지 확인
        volume: parseInt(values[5], 10),
      };
      return parsedRow;
    });
    return parsedData;
  };

  return (
    <Container>
      <MarketBox>
        <StockMarketMap stockData={marketData} isPrediction={false} />
        {loadingPredictionData ? (
          <p>Loading prediction data...</p>
        ) : (
          <StockMarketMap stockData={predictionData} isPrediction={true} />
        )}
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

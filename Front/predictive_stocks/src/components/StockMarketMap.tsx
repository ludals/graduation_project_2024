import React from "react";
import { Treemap, Tooltip, ResponsiveContainer } from "recharts";
import { proportionalSizes } from "./predefinedSizes";
import styled from "styled-components";

function getColor(change) {
  return change > 0 ? "#FF0000" : "#0000FF";
}

function parseCompanyName(name) {
  const parts = name.split("_");
  return parts[1];
}

export default function StockMarketMap({ stockData, isPrediction }) {
  const sortedData = stockData
    .map((stock) => ({
      name: parseCompanyName(stock.name),
      size: proportionalSizes[stock.name],
      change: stock.change,
      fill: getColor(stock.change),
    }))
    .sort((a, b) => b.size - a.size);

  return (
    <div style={{ width: "100%" , height: 800 }}>
      <Title>{isPrediction ? "Predicted Stock Market Map" : "Actual Stock Market Map"}</Title>
      <ResponsiveContainer>
        <Treemap
          data={sortedData}
          dataKey="size"
          nameKey="name"
          stroke="#fff"
          fill="#8884d8"
          content={<CustomizedContent />}
          
        >
          <Tooltip />
        </Treemap>
      </ResponsiveContainer>
    </div>
  );
}

const CustomizedContent = (props) => {
  const { x, y, width, height, name, change } = props;
  const formattedChange = typeof change === 'number' ? change.toFixed(2) : '0.00';

  return (
    <g>
      <rect
        x={x}
        y={y}
        width={width}
        height={height}
        style={{
          fill: getColor(change),
          stroke: "#fff",
          strokeWidth: 2,
          strokeOpacity: 1,
        }}
      />
      {width > 30 && height > 30 && (
        <>
          <text
            x={x + width / 2}
            y={y + height / 2 - 10}
            fill="#fff"
            fontSize={10} 
            textAnchor="middle"
            dominantBaseline="middle"
          >
            {name}
          </text>
          <text
            x={x + width / 2}
            y={y + height / 2 + 10}
            fill="#fff"
            fontSize={10}
            textAnchor="middle"
            dominantBaseline="middle"
          >
            {change > 0 ? `+${formattedChange}%` : `${formattedChange}%`}
          </text>
        </>
      )}
    </g>
  );
};

const Title = styled.h1`
  font-size: 28px;
  font-weight: 700;
  color: #333;
  text-align: center;
  margin: 20px 0;
  padding: 10px;
  border-bottom: 2px solid #ddd;
  background-color: #f9f9f9;
  border-radius: 4px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  font-family: 'Roboto', sans-serif;
`;
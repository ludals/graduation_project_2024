import React from "react";
import { Treemap, Tooltip, ResponsiveContainer } from "recharts";
import { proportionalSizes } from "./predefinedSizes";
import styled from "styled-components";
import { useRouter } from "next/router";

function getColor(change) {
  if (change >= 4) {
    return "#df3d43";
  } else if (change >= 2) {
    return "#ae4349";
  } else if (change > 0) {
    return "#81454f";
  } else if (Math.abs(change) < 0.01) {
    return "#424653";
  } else if (change >= -2) {
    return "#4861be";
  } else if (change >= -3) {
    return "#5174e8";
  } else {
    return "#5986f7";
  }
}

export default function StockMarketMap({ stockData, isPrediction }) {
  const router = useRouter();

  const handleClick = (name) => {
    router.push(`/stocks/${name}`);
  };

  const sortedData = stockData
    .map((stock) => ({
      name: stock.name,
      size: proportionalSizes[stock.name],
      change: stock.change,
      fill: getColor(stock.change),
    }))
    .sort((a, b) => b.size - a.size);

  return (
    <MarketMapContainer>
      <Title>
        {isPrediction
          ? "Predicted Stock Market Map (D+1)"
          : "Actual Stock Market Map (D-Day)"}
      </Title>
      <ResponsiveContainer width="100%" height="100%">
        <Treemap
          data={sortedData}
          dataKey="size"
          nameKey="name"
          stroke="#fff"
          fill="#8884d8"
          content={<CustomizedContent onClick={handleClick} />}
        >
          <Tooltip />
        </Treemap>
      </ResponsiveContainer>
    </MarketMapContainer>
  );
}

const CustomizedContent = (props) => {
  const { x, y, width, height, name, change, onClick } = props;
  const formattedChange =
    typeof change === "number" ? change.toFixed(2) : "0.00";
  const fontSize = Math.min(width, height) / 8;
  const textSpacing = Math.min(height / 13);
  return (
    <g onClick={() => onClick(name)} style={{ cursor: "pointer" }}>
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
            y={y + height / 2 - textSpacing}
            fill="#fff"
            fontSize={fontSize}
            textAnchor="middle"
            dominantBaseline="middle"
          >
            {name}
          </text>
          <text
            x={x + width / 2}
            y={y + height / 2 + textSpacing}
            fill="#fff"
            fontSize={fontSize}
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
  border-bottom: 2px solid #ddd;
  background-color: #f9f9f9;
  border-radius: 4px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  font-family: "Roboto", sans-serif;
`;

const MarketMapContainer = styled.div`
  display: flex;
  flex-direction: column;
  width: 100%;
  height: 100%;
`;

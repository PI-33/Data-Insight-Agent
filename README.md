# L'Oréal 数据洞察 Agent (Data Insight Agent)

一个基于 **ReAct Agent 架构** 的智能数据分析系统，借鉴 [Microsoft AutoGen](https://github.com/microsoft/autogen) 的多工具编排思想，让 AI 自主规划并执行多步骤数据分析。

[![在 ModelScope 中体验](https://img.shields.io/badge/在_ModelScope_中体验-blue)](https://www.modelscope.cn/studios/Pi33ymym/Loreal_Insight_Agent)

## 架构概览

```
用户查询
   │
   ▼
┌──────────────────────────┐
│   DataAnalysisAgent      │  ReAct 编排器
│   (plan → execute → sum) │
└──────────┬───────────────┘
           │ 调用
    ┌──────┴──────────────────────────────────┐
    │              ToolRegistry               │
    │  ┌──────────┐  ┌──────────────────────┐ │
    │  │ Inspector│  │   SQL Query Tool     │ │
    │  └──────────┘  └──────────────────────┘ │
    │  ┌──────────┐  ┌──────────────────────┐ │
    │  │Visualize │  │ Statistical Analysis │ │
    │  └──────────┘  └──────────────────────┘ │
    │  ┌──────────┐  ┌──────────────────────┐ │
    │  │Profiling │  │  Report Generator    │ │
    │  └──────────┘  └──────────────────────┘ │
    └─────────────────────────────────────────┘
```

## 功能特点

### 六大分析工具

| 工具 | 功能 | 场景 |
|------|------|------|
| **DataInspectorTool** | 表结构检查、字段解读、数据概览 | 了解数据长什么样 |
| **SQLQueryTool** | 自然语言→SQL 查询 | 精准数据检索 |
| **DataVisualizationTool** | 折线/柱状/饼图/散点/直方/热力图 | 数据可视化 |
| **StatisticalAnalysisTool** | 描述统计、相关性、趋势、Top-N | 深度统计分析 |
| **DataProfilingTool** | 缺失值/异常值/分布/质量评估 | 数据质量画像 |
| **ReportGeneratorTool** | 综合洞察报告生成 | 自动报告 |

### Agent 能力

- **自主规划**: Agent 根据用户查询自动选择合适的工具组合
- **多步骤执行**: 按依赖关系顺序执行多个工具
- **结果综合**: 汇总所有工具结果，生成专业分析报告
- **对话记忆**: 支持多轮对话，上下文感知
- **优雅降级**: 工具失败时自动回退

## 项目结构

```
├── app.py                      # Gradio Web 入口
├── agent/
│   ├── data_agent.py           # ReAct Agent 编排器
│   └── tool_registry.py        # 工具注册管理
├── tools/
│   ├── base_tool.py            # 工具基类
│   ├── data_inspector.py       # 数据检查工具
│   ├── sql_query.py            # SQL 查询工具
│   ├── data_visualization.py   # 数据可视化工具
│   ├── statistical_analysis.py # 统计分析工具
│   ├── data_profiling.py       # 数据画像工具
│   └── report_generator.py     # 报告生成工具
├── core/
│   ├── llm_client.py           # LLM 客户端（支持函数调用）
│   ├── database.py             # 数据库管理器
│   └── dialogue_context.py     # 对话上下文管理
├── utils/
│   └── logger.py               # 日志工具
├── data/
│   ├── data.csv                # 源数据
│   └── order_database.db       # SQLite 数据库
├── import_csv_to_sqlite.py     # 数据导入脚本
├── requirements.txt
└── setup.py
```

## 快速开始

### 环境要求

- Python >= 3.10
- SQLite3

### 安装

```bash
git clone https://github.com/PI-33/Data-Insight-Agent.git
cd Data-Insight-Agent

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 配置

创建 `.env` 文件：

```
API_KEY=your_api_key_here
BASE_URL=https://api.siliconflow.cn/v1
```

### 导入数据（首次运行）

```bash
python import_csv_to_sqlite.py
```

### 启动

```bash
python app.py
```

访问 http://localhost:7860

### Docker 部署

```bash
docker-compose up -d
```

## 使用示例

### 数据探查
```
帮我查看一下数据库有哪些表和字段
分析一下数据整体质量如何
```

### 数据查询
```
统计每个产品在10月份的销售总额和销售数量
查询一线城市的销售情况
```

### 可视化分析
```
绘制2024年10月21日到10月30日的每日销售额趋势图
展示各销售渠道的销售额占比饼图
展示销售额前15的城市销售情况柱状图
```

### 深度分析
```
分析各省份的销售额排名
对销售数据做全面的统计分析并生成报告
```

## 在线体验

访问 [ModelScope 体验页面](https://www.modelscope.cn/studios/Pi33ymym/Loreal_Insight_Agent) 立即开始使用。

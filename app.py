"""
L'Oréal Data Insight Agent – Gradio UI

A multi-tool AI agent for comprehensive data analysis, powered by a
ReAct-style orchestrator inspired by Microsoft AutoGen.
"""

import logging
import os

import gradio as gr

from agent.data_agent import DataAnalysisAgent
from utils.logger import setup_logging

# Initialise agent
agent = DataAnalysisAgent()


# ------------------------------------------------------------------
# Gradio interface
# ------------------------------------------------------------------
def create_interface():
    with gr.Blocks(
        title="L'Oréal 数据洞察 Agent",
        theme=gr.themes.Soft(),
        css=open("static/css/style.css").read(),
    ) as interface:

        # Header
        gr.HTML("""
        <div class="main-header">
            <h1 style="margin:0;font-size:2rem;font-weight:700;">
                L'Oréal 数据洞察 Agent
            </h1>
            <p style="margin:0.5rem 0 0;font-size:1rem;opacity:0.9;">
                多工具智能数据分析 · 由 ReAct Agent 驱动
            </p>
        </div>
        """)

        # Tool cards
        with gr.Row():
            for icon, title, desc in [
                ("🔍", "数据探查", "表结构 · 字段解读 · 数据概览"),
                ("📊", "智能可视化", "折线 · 柱状 · 饼图 · 散点 · 热力图"),
                ("📈", "统计分析", "描述统计 · 趋势 · 相关性 · Top-N"),
                ("🧪", "数据画像", "缺失值 · 异常值 · 分布 · 质量评估"),
                ("📝", "分析报告", "自动综合洞察 · 业务建议"),
            ]:
                with gr.Column(scale=1, min_width=120):
                    gr.HTML(f"""
                    <div class="feature-card" style="text-align:center;">
                        <div style="font-size:1.6rem;">{icon}</div>
                        <h4 style="margin:0.3rem 0;font-size:0.95rem;">{title}</h4>
                        <p style="margin:0;font-size:0.78rem;opacity:0.8;">{desc}</p>
                    </div>
                    """)

        # Chat + detail panel
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    height=480,
                    label="对话",
                    type="messages",
                    show_copy_button=True,
                )
                with gr.Group():
                    msg = gr.Textbox(
                        placeholder="请输入您的数据分析问题…",
                        show_label=False,
                        lines=2,
                        max_lines=4,
                        container=False,
                    )
                    with gr.Row():
                        submit_btn = gr.Button("发送", variant="primary", scale=2)
                        clear_btn = gr.Button("清空对话", variant="secondary", scale=1)

            with gr.Column(scale=1, min_width=260):
                with gr.Accordion("Agent 工作台", open=True):
                    status_display = gr.Markdown("等待查询…")
                with gr.Accordion("工具调用日志", open=False):
                    tool_log = gr.Textbox(
                        label="",
                        lines=12,
                        interactive=False,
                        show_label=False,
                    )
                with gr.Accordion("已注册工具", open=False):
                    tools_info = gr.Markdown(
                        _format_tools_info(agent.get_available_tools())
                    )

        # Example queries
        with gr.Row():
            with gr.Column(scale=1):
                gr.HTML("<h4 style='color:var(--loreal-gold);'>🔍 数据探查</h4>")
                gr.Examples(
                    [
                        "帮我查看一下数据库有哪些表和字段",
                        "分析一下数据整体质量如何",
                    ],
                    inputs=msg,
                )
            with gr.Column(scale=1):
                gr.HTML("<h4 style='color:var(--loreal-gold);'>📊 可视化分析</h4>")
                gr.Examples(
                    [
                        "绘制2024年10月21日到10月30日的每日销售额趋势图",
                        "展示各销售渠道的销售额占比饼图",
                        "展示销售额前15的城市销售情况柱状图",
                    ],
                    inputs=msg,
                )
            with gr.Column(scale=1):
                gr.HTML("<h4 style='color:var(--loreal-gold);'>📈 深度分析</h4>")
                gr.Examples(
                    [
                        "统计每个产品在10月份的销售总额和销售数量",
                        "分析各省份的销售额排名和同比情况",
                        "对销售数据做全面的统计分析并生成报告",
                    ],
                    inputs=msg,
                )

        # ---- callbacks ------------------------------------------------
        def on_user_submit(user_message, history):
            history = history or []
            history.append({"role": "user", "content": user_message})
            return "", history

        def on_bot_response(history):
            user_message = history[-1]["content"]
            status = "**正在分析…** 🔄\n\nAgent 正在规划并执行工具链…"

            yield history, status, ""

            result = agent.process_query(user_message)
            response = result["response"]
            images = result.get("images", [])
            tool_results = result.get("tool_results", [])

            log_lines = []
            for i, tr in enumerate(tool_results, 1):
                success = "✅" if tr.get("success") else "❌"
                log_lines.append(
                    f"[{i}] {success} {tr.get('tool', '?')} — {tr.get('reason', '')}"
                )
            log_text = "\n".join(log_lines) if log_lines else "无工具调用"

            history.append({"role": "assistant", "content": response})

            for img in images:
                if img and os.path.exists(img):
                    history.append({"role": "assistant", "content": {"path": img}})

            tools_used = [tr.get("tool") for tr in tool_results]
            status = (
                f"**分析完成** ✅\n\n"
                f"- 调用工具 {len(tool_results)} 个\n"
                f"- 工具链: {' → '.join(tools_used) if tools_used else '无'}\n"
                f"- 生成图表 {len(images)} 张"
            )
            yield history, status, log_text

        def on_clear():
            agent.clear_context()
            return [], "", "等待查询…", ""

        msg.submit(on_user_submit, [msg, chatbot], [msg, chatbot], queue=False).then(
            on_bot_response, chatbot, [chatbot, status_display, tool_log]
        )
        submit_btn.click(on_user_submit, [msg, chatbot], [msg, chatbot], queue=False).then(
            on_bot_response, chatbot, [chatbot, status_display, tool_log]
        )
        clear_btn.click(
            on_clear, outputs=[chatbot, msg, status_display, tool_log]
        )

    return interface


def _format_tools_info(tools):
    lines = []
    for t in tools:
        lines.append(f"**{t['name']}**\n{t['description']}\n")
    return "\n".join(lines)


# ------------------------------------------------------------------
def main():
    setup_logging()
    logging.info("=== L'Oréal Data Insight Agent 启动 ===")
    interface = create_interface()
    interface.launch(share=False)


if __name__ == "__main__":
    main()

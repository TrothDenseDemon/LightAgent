"""示例：多代理协作与会话管理"""

import os
from pprint import pprint

from LightAgent import LightAgent, LightSwarm

# Set Environment Variables OPENAI_API_KEY and OPENAI_BASE_URL
# The default model uses gpt-4o-mini
os.environ.setdefault("OPENAI_API_KEY", "your_api_key")
os.environ.setdefault("OPENAI_BASE_URL", "http://your_base_url/v1")


def collaborative_prompt(agent_name, last_message, session_state):
    """根据最近的对话为下一位代理构造提示，并记录共享记忆。"""

    notes = session_state.shared_state.setdefault("notes", [])
    if last_message and last_message.get("content"):
        record = f"{last_message.get('name', last_message.get('role'))}: {last_message['content']}"
        if not notes or notes[-1] != record:
            notes.append(record)

    summary = "\n".join(notes[-3:]) if notes else "暂无共享记忆。"
    base_task = session_state.shared_state.get("task", "")
    return (
        f"协作任务：{base_task}\n\n"
        f"最近的协作摘要：\n{summary}\n\n"
        f"轮到 {agent_name} 发言，请结合上下文给出明确的行动或结论。"
    )


def auto_finish(session_state, _swarm):
    """当任一代理输出 FINAL_ANSWER 时提前结束会话。"""

    if not session_state.history:
        return False
    latest = session_state.history[-1]
    content = latest.get("content")
    return isinstance(content, str) and "FINAL_ANSWER" in content


# Create an instance of LightSwarm
light_swarm = LightSwarm()

# Create multiple agents with distinct roles
planner = LightAgent(
    name="Planner",
    instructions="汇总用户需求并拆解为执行步骤。",
    role="planner",
    model="gpt-4o-mini",
)

builder = LightAgent(
    name="Builder",
    instructions="根据规划产出技术实现方案或伪代码。",
    role="executor",
    model="gpt-4o-mini",
)

reviewer = LightAgent(
    name="Reviewer",
    instructions="评审方案并补充风险提示，必要时给出最终答复（包含 FINAL_ANSWER 标记）。",
    role="reviewer",
    model="gpt-4o-mini",
)

# Automatically register agents to the LightSwarm instance
light_swarm.register_agent(planner, builder, reviewer)

# Create a session with routing rules and shared memory
session = light_swarm.create_session(
    participants=[planner, builder, reviewer],
    max_rounds=6,
    routing_strategy="role",
    routing_rules={
        "role_transitions": {
            "planner": "Builder",
            "executor": "Reviewer",
            "reviewer": "Planner",
        },
        "fallback": "Planner",
    },
    shared_state={
        "task": "帮助用户设计一个带有健康检查与推荐接口的 FastAPI 服务",
        "notes": [],
    },
    prompt_builder=collaborative_prompt,
    termination_condition=auto_finish,
    auto_stop_tokens=["FINAL_ANSWER"],
)

# Run a collaborative conversation
session = light_swarm.run_group_chat(
    session=session,
    initial_prompt="用户：请帮我规划一个FastAPI服务，提供 /health 与 /recommend 接口。",
    user_name="用户",
)

print("=== 群聊历史 ===")
for message in session.history:
    speaker = message.get("name", message.get("role"))
    print(f"[{speaker}] {message.get('content')}")

print("\n共享记忆片段：")
pprint(session.shared_state.get("notes", []))

# Manual stop example - downstream调度器可通过 shared_state 控制流程
light_swarm.update_shared_state(session.session_id, stop=True)
light_swarm.run_group_chat(session=session)
print(f"手动终止标志：{session.shared_state.get('stop')}")
light_swarm.update_shared_state(session.session_id, stop=False)

# Execute a task graph where agents reuse the accumulated context
tasks = [
    {
        "id": "plan",
        "agent": "Planner",
        "prompt": "结合已有讨论，列出三条实施步骤。",
    },
    {
        "id": "build",
        "agent": "Builder",
        "depends_on": ["plan"],
        "prompt": lambda results, _: (
            "根据以下步骤给出伪代码：\n" + results["plan"]
        ),
    },
    {
        "id": "review",
        "agent": "Reviewer",
        "depends_on": ["build"],
        "prompt": lambda results, _: (
            "评估伪代码的风险并给出改进建议，最后输出 FINAL_ANSWER=OK。\n" + results["build"]
        ),
    },
]

graph_results = light_swarm.run_task_graph(tasks, session=session)

print("\n=== 任务图结果 ===")
pprint(graph_results)

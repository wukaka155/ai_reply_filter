"""
AI智能回复过滤器插件 (AI Reply Filter)

通过调用AI模型分析每条消息,智能判断是否需要回复。
支持私聊/群聊独立配置,可设置群组白名单/黑名单过滤。
**v1.4新增**: 消息合并等待模式,智能收集连续消息!

## 主要功能

- **AI智能判断**: 使用AI模型分析消息内容,决定是否需要回复
- **消息合并等待**: 等待一段时间收集用户连续发送的多条消息,合并处理提升理解准确度
- **完全接管模式**: 可选择完全接管回复控制,阻止不需要回复的消息触发随机回复
- **自动人设识别**: 自动从数据库读取频道配置的人设,无需手动配置
- **智能上下文感知**: 自动获取历史聊天记录,理解对话连贯性
- **私聊/群聊独立控制**: 可分别为私聊和群聊设置不同的过滤规则
- **群组过滤**: 支持群组白名单和黑名单,精确控制生效范围
- **自定义提示词**: 可自定义AI判断的系统提示词
- **智能缓存**: 自动缓存AI判断结果,减少重复调用

## 配置说明

### 1. 基础配置
- **启用私聊过滤**: 是否在私聊频道启用AI过滤
- **启用群聊过滤**: 是否在群聊频道启用AI过滤
- **完全接管模式**: 启用后完全阻止不需要回复的消息,包括随机回复
- **启用消息合并等待**: 等待收集连续消息后合并处理
- **消息合并等待时间**: 等待收集消息的时间(秒)
- **最大合并消息数**: 单次最多合并的消息数量
- **AI分析模型组**: 用于消息分析的AI模型组

### 2. 群组过滤配置
- **群组过滤模式**: 禁用/白名单/黑名单
- **群组ID列表**: 白名单或黑名单的群组ID列表

### 3. AI判断配置
- **自动使用频道人设**: 是否自动读取频道配置的人设(推荐开启)
- **上下文消息数量**: 判断时包含的历史消息数量(建议5-10条)
- **系统提示词**: 指导AI如何判断消息是否需要回复

## 使用场景

### 场景1: 智能过滤无意义消息
- AI自动判断"嗯"、"好的"等简短消息不需要回复
- 只对有实质内容的消息进行回复

### 场景2: 根据人设过滤
- 插件自动读取频道配置的人设
- AI根据人设判断消息是否符合回复范围
- 例如:编程助手人设只回复编程相关问题

### 场景3: 群组精准控制
- 只在指定的几个群组中启用AI过滤
- 排除某些测试群组或不需要过滤的群组

### 场景4: 上下文相关性判断
- AI自动读取最近的对话历史
- 判断消息是否与当前对话主题相关
- 过滤掉无关的插话或闲聊

### 场景5: 完全接管回复控制
- 启用完全接管模式后,AI判断不需要回复的消息将完全阻止
- 阻止随机回复触发,只在AI明确判断需要回复时才响应
- 适合需要精确控制回复时机的场景

### 场景6: 消息合并处理
- 用户经常连续发送多条消息表达完整意思
- 启用消息合并后,等待用户发完再统一处理
- AI能看到完整上下文,提供更准确的回复

## 工作原理

1. **消息拦截**: 通过用户消息回调拦截所有消息
2. **范围检查**: 检查频道类型和群组是否在过滤范围内
3. **人设读取**: 自动从数据库读取频道关联的人设(如果有)
4. **上下文获取**: 自动获取最近N条聊天记录作为上下文
5. **AI分析**: 调用AI模型分析消息内容
6. **决策执行**: 根据AI返回结果和接管模式决定是否允许触发回复
"""

import json
import re
import time
from typing import Dict, List, Optional, Literal

from pydantic import Field

from nekro_agent.api import core
from nekro_agent.api.message import ChatMessage
from nekro_agent.api.schemas import AgentCtx
from nekro_agent.api.signal import MsgSignal
from nekro_agent.models.db_chat_message import DBChatMessage
from nekro_agent.models.db_chat_channel import DBChatChannel
from nekro_agent.models.db_preset import DBPreset
from nekro_agent.services.agent.openai import gen_openai_chat_response
from nekro_agent.services.plugin.base import NekroPlugin, ConfigBase

# 创建插件实例
plugin = NekroPlugin(
    name="AI自主判断是否回复",
    module_name="ai_reply_filter",
    description="通过AI模型智能分析消息内容,判断是否需要触发回复。支持自动读取频道人设和聊天记录,私聊/群聊独立配置,群组过滤,完全接管模式,消息合并等待。",
    version="1.5.0",
    author="xiaojiu",
    url="https://github.com/miuzhaii/ai_reply_filter",
    support_adapter=["onebot_v11", "discord", "telegram", "wechatpad", "wxwork"],
)


# 配置定义
@plugin.mount_config()
class AIReplyFilterConfig(ConfigBase):
    """AI回复过滤器配置"""

    # 基础开关配置
    ENABLE_PRIVATE: bool = Field(
        default=True,
        title="启用私聊过滤",
        description="是否在私聊频道启用AI智能过滤功能",
    )

    ENABLE_GROUP: bool = Field(
        default=True,
        title="启用群聊过滤",
        description="是否在群聊频道启用AI智能过滤功能",
    )

    ENABLE_COMPLETE_TAKEOVER: bool = Field(
        default=False,
        title="完全接管模式",
        description="启用后,AI判断不需要回复的消息将完全阻止触发(包括随机回复)。禁用时,只处理主动触发的情况,不影响随机回复。",
    )

    # AI模型配置
    AI_MODEL_GROUP: str = Field(
        default="default",
        title="AI分析模型组",
        description="用于消息分析的AI模型组名称",
        json_schema_extra={"ref_model_groups": True, "required": True, "model_type": "chat"},
    )

    # 群组过滤配置
    GROUP_FILTER_MODE: Literal["禁用", "白名单", "黑名单"] = Field(
        default="禁用",
        title="群组过滤模式",
        description="禁用=所有群组都应用过滤，白名单=仅列表中的群组生效，黑名单=排除列表中的群组",
    )

    GROUP_ID_LIST: list[str] = Field(
        default=[],
        title="群组ID列表",
        description="群组ID列表，用于白名单或黑名单过滤。群组ID格式通常为: group_群号",
    )

    # AI判断配置
    AUTO_USE_PRESET: bool = Field(
        default=True,
        title="自动使用频道人设",
        description="是否自动从数据库读取当前频道关联的人设信息。启用后，AI会根据频道配置的人设来判断是否需要回复。",
    )

    SYSTEM_PROMPT: str = Field(
        default="""判断消息是否需要AI回复。

规则：问题/请求→true，简单问候/确认→false，@AI→true

返回JSON: {"should_reply": true} 或 {"should_reply": false}""",
        title="AI判断系统提示词",
        description="指导AI如何判断消息是否需要回复的系统提示词",
    )

    CONTEXT_MESSAGE_COUNT: int = Field(
        default=5,
        title="上下文消息数量",
        description="判断时包含的历史消息数量。0=不使用上下文，1-20=包含最近N条消息。建议值：5-10条。带上上下文可以让AI更好地理解对话连贯性。",
    )

    # 消息合并配置
    ENABLE_MESSAGE_MERGE: bool = Field(
        default=False,
        title="启用消息合并等待",
        description="启用后,触发回复时会等待一段时间收集后续消息再处理。适合用户连发多条消息的场景。",
    )

    MESSAGE_MERGE_WAIT_TIME: float = Field(
        default=10.0,
        title="消息合并等待时间(秒)",
        description="触发回复后等待的时间,用于收集后续消息。建议5-15秒。",
    )

    MESSAGE_MERGE_MAX_COUNT: int = Field(
        default=5,
        title="最大合并消息数",
        description="单次最多合并的消息数量,达到后立即处理。0表示不限制。",
    )



# 获取配置和插件存储
config: AIReplyFilterConfig = plugin.get_config(AIReplyFilterConfig)
store = plugin.store

# 硬编码的常量配置（技术细节，不需要用户配置）
CACHE_EXPIRE_SECONDS = 300  # 缓存过期时间：5分钟
AI_TIMEOUT = 10  # AI判断超时时间：10秒
BLOCK_MODE = 1  # 阻止模式：1=阻止AI响应但保存记录

# 消息合并状态管理
merge_tasks: Dict[str, dict] = {}  # chat_key -> {messages, ctx, task, created_at}
merge_locks: Dict[str, "asyncio.Lock"] = {}  # chat_key -> Lock


# region: 缓存管理

async def get_cached_decision(message_hash: str) -> Optional[bool]:
    """获取缓存的AI判断结果"""
    cache_key = f"ai_decision_{message_hash}"
    cached_data = await store.get(store_key=cache_key)

    if not cached_data:
        return None

    try:
        data = json.loads(cached_data)
        timestamp = data.get("timestamp", 0)
        decision = data.get("decision")

        # 检查缓存是否过期
        if time.time() - timestamp > CACHE_EXPIRE_SECONDS:
            core.logger.debug(f"[AI回复过滤器] 缓存已过期: {cache_key}")
            return None

        core.logger.debug(f"[AI回复过滤器] 使用缓存结果: {decision}")
        return decision
    except Exception as e:
        core.logger.error(f"[AI回复过滤器] 解析缓存数据失败: {e}")
        return None


async def save_decision_to_cache(message_hash: str, decision: bool):
    """保存AI判断结果到缓存"""
    cache_key = f"ai_decision_{message_hash}"
    cache_data = {
        "decision": decision,
        "timestamp": time.time(),
    }

    await store.set(store_key=cache_key, value=json.dumps(cache_data))
    core.logger.debug(f"[AI回复过滤器] 已缓存判断结果: {cache_key} = {decision}")


# endregion: 缓存管理


# region: 上下文消息获取

async def get_context_messages(chat_key: str, limit: int) -> List[DBChatMessage]:
    """
    获取历史消息上下文

    Args:
        chat_key: 聊天频道标识
        limit: 获取的消息数量

    Returns:
        List[DBChatMessage]: 历史消息列表，按时间倒序
    """
    if limit <= 0:
        return []

    try:
        # 查询最近的消息（不包括当前消息）
        messages = (
            await DBChatMessage.filter(
                chat_key=chat_key,
                is_recalled=False,
            )
            .order_by("-send_timestamp")
            .limit(limit)
        )

        core.logger.debug(f"[AI回复过滤器] 获取到 {len(messages)} 条上下文消息")
        return messages

    except Exception as e:
        core.logger.error(f"[AI回复过滤器] 获取上下文消息失败: {e}")
        return []


def format_context_messages(messages: List[DBChatMessage]) -> str:
    """
    格式化上下文消息为可读文本

    Args:
        messages: 消息列表

    Returns:
        str: 格式化后的上下文文本
    """
    if not messages:
        return ""

    # 消息按时间倒序，需要反转为正序显示
    messages = list(reversed(messages))

    context_lines = []
    for msg in messages:
        # 获取发送者昵称，如果没有则使用用户ID
        sender = msg.sender_nickname if msg.sender_nickname else msg.platform_userid
        # 清理消息内容
        content = msg.content_text.strip()
        if content:
            context_lines.append(f"{sender}: {content}")

    return "\n".join(context_lines)


# endregion: 上下文消息获取


# region: 人设获取

async def get_channel_preset(chat_key: str) -> Optional[DBPreset]:
    """
    获取频道关联的人设信息

    Args:
        chat_key: 聊天频道标识

    Returns:
        Optional[DBPreset]: 人设对象，如果未找到则返回 None
    """
    if not config.AUTO_USE_PRESET:
        return None

    try:
        # 获取频道信息
        channel = await DBChatChannel.get_or_none(chat_key=chat_key)
        if not channel or not channel.preset_id:
            core.logger.debug(f"[AI回复过滤器] 频道 {chat_key} 未配置人设")
            return None

        # 获取人设信息
        preset = await DBPreset.get_or_none(id=channel.preset_id)
        if preset:
            core.logger.info(f"[AI回复过滤器] 使用频道人设: {preset.name}")
        else:
            core.logger.warning(f"[AI回复过滤器] 人设ID {channel.preset_id} 不存在")

        return preset

    except Exception as e:
        core.logger.error(f"[AI回复过滤器] 获取频道人设失败: {e}")
        return None


# endregion: 人设获取


# region: AI判断逻辑

async def ai_should_reply(message_text: str, chat_key: str = "") -> bool:
    """
    使用AI判断是否应该回复消息

    Args:
        message_text: 消息内容
        chat_key: 聊天频道标识（用于获取历史消息）

    Returns:
        bool: True表示应该回复，False表示不需要回复
    """
    # 生成消息hash用于缓存
    import hashlib
    message_hash = hashlib.md5(message_text.encode()).hexdigest()

    # 尝试从缓存获取
    cached_decision = await get_cached_decision(message_hash)
    if cached_decision is not None:
        return cached_decision

    try:
        # 获取模型组配置
        model_group = core.config.get_model_group_info(config.AI_MODEL_GROUP)

        # 调用AI进行判断
        core.logger.info(f"[AI回复过滤器] 调用AI分析消息: {message_text[:50]}...")

        # 获取频道人设（如果启用）
        preset = None
        preset_text = ""
        if chat_key:
            preset = await get_channel_preset(chat_key)
            if preset:
                preset_text = f"{preset.content}"
                core.logger.debug(f"[AI回复过滤器] 使用频道人设: {preset.name}")

        # 获取历史消息上下文
        context_text = ""
        if config.CONTEXT_MESSAGE_COUNT > 0 and chat_key:
            context_messages = await get_context_messages(chat_key, config.CONTEXT_MESSAGE_COUNT)
            if context_messages:
                context_text = format_context_messages(context_messages)
                core.logger.debug(f"[AI回复过滤器] 使用 {len(context_messages)} 条历史消息作为上下文")

        # 构建用户消息内容
        user_message_parts = []

        # 1. 添加人设信息（如果有）
        if preset_text:
            user_message_parts.append(f"AI助手人设信息:\n{preset_text}")
            core.logger.debug(f"[AI回复过滤器] 使用自动获取的人设进行判断")

        # 2. 添加历史消息上下文（如果有）
        if context_text:
            user_message_parts.append(f"近期对话历史:\n{context_text}")

        # 3. 添加当前消息
        user_message_parts.append(f"当前消息: {message_text}")

        # 4. 添加判断提示
        if preset_text or context_text:
            prompt_parts = []
            if preset_text:
                prompt_parts.append("人设信息")
            if context_text:
                prompt_parts.append("对话上下文")
            user_message_parts.append(f"\n请根据上述{' 和 '.join(prompt_parts)}，判断当前消息是否需要AI回复。")

        user_message = "\n\n".join(user_message_parts)

        response = await gen_openai_chat_response(
            model=model_group.CHAT_MODEL,
            messages=[
                {"role": "system", "content": config.SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            base_url=model_group.BASE_URL,
            api_key=model_group.API_KEY,
            temperature=0.3,
            max_tokens=32000,
        )

        content = response.response_content.strip()
        core.logger.debug(f"[AI回复过滤器] AI原始返回: {content}")

        # 解析JSON响应
        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            should_reply = result.get("should_reply", True)
        else:
            # 如果无法解析JSON，默认允许回复
            core.logger.warning(f"[AI回复过滤器] 无法解析AI返回的JSON，默认允许回复")
            should_reply = True

        # 保存到缓存
        await save_decision_to_cache(message_hash, should_reply)

        core.logger.info(f"[AI回复过滤器] AI判断结果: {'需要回复' if should_reply else '不需要回复'}")
        return should_reply

    except Exception as e:
        # AI调用失败时，默认允许回复，避免阻塞正常对话
        core.logger.error(f"[AI回复过滤器] AI判断失败: {e}", exc_info=True)
        return True


# endregion: AI判断逻辑


# region: 群组过滤逻辑

def should_filter_group(channel_id: str) -> bool:
    """
    判断群组是否需要应用过滤

    Args:
        channel_id: 频道ID (可能是 "group_123456" 或 "123456" 格式)

    Returns:
        bool: True表示需要过滤，False表示不过滤
    """
    if config.GROUP_FILTER_MODE == "禁用":
        # 禁用群组过滤，所有群组都过滤
        return True

    # 智能匹配：支持带前缀和不带前缀的群组ID
    # 例如：channel_id="group_1067597714" 能匹配配置中的 "1067597714" 或 "group_1067597714"
    is_in_list = False
    
    # 提取不带前缀的群号
    channel_id_without_prefix = channel_id.replace("group_", "").replace("private_", "")
    
    core.logger.info(f"[AI回复过滤器] 检查群组: {channel_id} (纯ID: {channel_id_without_prefix})")
    core.logger.info(f"[AI回复过滤器] 配置的群组列表: {config.GROUP_ID_LIST}")
    core.logger.info(f"[AI回复过滤器] 过滤模式: {config.GROUP_FILTER_MODE}")
    
    for group_id in config.GROUP_ID_LIST:
        # 清理配置中的群组ID
        config_id_without_prefix = str(group_id).replace("group_", "").replace("private_", "")
        
        core.logger.info(f"[AI回复过滤器] 比对: '{channel_id_without_prefix}' vs '{config_id_without_prefix}'")
        
        # 匹配：只要群号相同就算匹配
        if channel_id_without_prefix == config_id_without_prefix:
            is_in_list = True
            core.logger.info(f"[AI回复过滤器] ✅ 群组ID匹配成功: {channel_id} <-> {group_id}")
            break
    
    if not is_in_list:
        core.logger.info(f"[AI回复过滤器] ❌ 群组ID未匹配到任何配置")

    if config.GROUP_FILTER_MODE == "白名单":
        # 白名单模式：只有在列表中的群组才过滤
        result = is_in_list
        core.logger.info(f"[AI回复过滤器] 白名单模式判断结果: {'需要过滤' if result else '不需要过滤'}")
        return result
    elif config.GROUP_FILTER_MODE == "黑名单":
        # 黑名单模式：排除列表中的群组
        result = not is_in_list
        core.logger.info(f"[AI回复过滤器] 黑名单模式判断结果: {'需要过滤' if result else '不需要过滤'}")
        return result

    return True


# endregion: 群组过滤逻辑


# region: 消息合并逻辑

import asyncio
from copy import copy


async def get_merge_lock(chat_key: str) -> asyncio.Lock:
    """获取或创建频道的锁"""
    if chat_key not in merge_locks:
        merge_locks[chat_key] = asyncio.Lock()
    return merge_locks[chat_key]


async def handle_with_merge(_ctx: AgentCtx, message: ChatMessage, chat_key: str) -> MsgSignal:
    """
    使用消息合并模式处理消息
    群聊模式下，收集所有用户的消息一起合并（作为完整对话），但会标注每条消息的发送者
    
    Args:
        _ctx: Agent上下文
        message: 用户消息
        chat_key: 聊天频道标识
    
    Returns:
        MsgSignal: 总是返回BLOCK_TRIGGER，实际回复由合并处理触发
    """
    lock = await get_merge_lock(chat_key)
    
    async with lock:
        if chat_key in merge_tasks:
            # 已有等待任务，将消息加入队列
            task_info = merge_tasks[chat_key]
            task_info['messages'].append(message)
            
            core.logger.info(f"[消息合并] 加入消息到等待队列，当前数量: {len(task_info['messages'])}")
            
            # 检查是否达到最大数量
            if config.MESSAGE_MERGE_MAX_COUNT > 0 and len(task_info['messages']) >= config.MESSAGE_MERGE_MAX_COUNT:
                core.logger.info(f"[消息合并] 达到最大数量 {config.MESSAGE_MERGE_MAX_COUNT}，立即处理")
                task_info['task'].cancel()
            
            return MsgSignal.BLOCK_TRIGGER
        else:
            # 创建新的等待任务
            core.logger.info(f"[消息合并] 创建新的等待任务，等待时间: {config.MESSAGE_MERGE_WAIT_TIME}秒")
            
            task_info = {
                'messages': [message],
                'ctx': _ctx,
                'created_at': time.time(),
            }
            
            # 启动定时器
            timer_task = asyncio.create_task(
                wait_and_process(chat_key)
            )
            task_info['task'] = timer_task
            merge_tasks[chat_key] = task_info
            
            return MsgSignal.BLOCK_TRIGGER


async def wait_and_process(chat_key: str):
    """等待指定时间后处理合并的消息"""
    try:
        core.logger.info(f"[消息合并] 启动等待计时器")
        await asyncio.sleep(config.MESSAGE_MERGE_WAIT_TIME)
        core.logger.info(f"[消息合并] 等待时间到，开始处理消息")
        await process_merged_messages(chat_key)
    except asyncio.CancelledError:
        core.logger.info(f"[消息合并] 计时器被取消，立即处理消息")
        await process_merged_messages(chat_key)
    except Exception as e:
        core.logger.error(f"[消息合并] 等待处理异常: {e}", exc_info=True)


async def process_merged_messages(chat_key: str):
    """处理合并的消息并进行AI判断"""
    lock = await get_merge_lock(chat_key)

    async with lock:
        if chat_key not in merge_tasks:
            return

        task_info = merge_tasks[chat_key]
        messages = task_info['messages']
        ctx = task_info['ctx']

        # 🔍 区分用户：为每条消息标注发送者信息
        merged_lines = []
        for msg in messages:
            # 获取发送者信息
            sender_name = getattr(msg, 'sender_nickname', None) or getattr(msg, 'platform_userid', 'unknown')
            sender_id = getattr(msg, 'sender_id', '') or getattr(msg, 'platform_userid', '')
            content = msg.content_text
            
            # 格式：[发送者昵称(ID)] 消息内容
            merged_lines.append(f"[{sender_name}({sender_id})] {content}")
        
        merged_content = "\n".join(merged_lines)

        core.logger.info(f"[消息合并] 合并了 {len(messages)} 条消息进行处理")
        core.logger.debug(f"[消息合并] 合并内容:\n{merged_content[:300]}...")

        # 清理任务
        del merge_tasks[chat_key]

        # 对合并后的内容进行AI判断
        should_reply = await ai_should_reply(merged_content, chat_key)

        if should_reply:
            # AI判断需要回复,触发AI响应
            core.logger.info(f"[消息合并] AI判断需要回复,触发响应")
            try:
                from nekro_agent.services.message_service import message_service

                await message_service.push_system_message(
                    chat_key=chat_key,
                    agent_messages=f"群内连续发送了 {len(messages)} 条消息,已合并:\n{merged_content}",
                    trigger_agent=True,
                )

                core.logger.info(f"[消息合并] 成功触发AI回复")
            except Exception as e:
                core.logger.error(f"[消息合并] 触发AI回复失败: {e}", exc_info=True)
        else:
            # AI判断不需要回复
            core.logger.info(f"[消息合并] AI判断不需要回复,忽略这批消息")


# endregion: 消息合并逻辑


# region: 插件初始化

@plugin.mount_init_method()
async def initialize_plugin():
    """插件初始化"""
    core.logger.info(f"插件 '{plugin.name}' 正在初始化...")

    # 验证配置
    if config.ENABLE_PRIVATE or config.ENABLE_GROUP:
        try:
            model_group = core.config.get_model_group_info(config.AI_MODEL_GROUP)
            core.logger.info(f"[AI回复过滤器] 使用模型组: {config.AI_MODEL_GROUP}")
            core.logger.info(f"[AI回复过滤器] 模型: {model_group.CHAT_MODEL}")
        except Exception as e:
            core.logger.error(f"[AI回复过滤器] 模型组配置错误: {e}")

    core.logger.info(f"[AI回复过滤器] 私聊过滤: {'启用' if config.ENABLE_PRIVATE else '禁用'}")
    core.logger.info(f"[AI回复过滤器] 群聊过滤: {'启用' if config.ENABLE_GROUP else '禁用'}")
    core.logger.info(f"[AI回复过滤器] 群组过滤模式: {config.GROUP_FILTER_MODE}")
    core.logger.info(f"[AI回复过滤器] 群组列表: {config.GROUP_ID_LIST}")
    core.logger.info(f"[AI回复过滤器] 完全接管模式: {'启用' if config.ENABLE_COMPLETE_TAKEOVER else '禁用'}")
    core.logger.info(f"[AI回复过滤器] 消息合并模式: {'启用' if config.ENABLE_MESSAGE_MERGE else '禁用'}")
    if config.ENABLE_MESSAGE_MERGE:
        core.logger.info(f"[AI回复过滤器] 消息合并等待时间: {config.MESSAGE_MERGE_WAIT_TIME}秒")
        core.logger.info(f"[AI回复过滤器] 最大合并消息数: {config.MESSAGE_MERGE_MAX_COUNT}")

    core.logger.success(f"插件 '{plugin.name}' 初始化完成。")


# endregion: 插件初始化


# region: 用户消息回调

@plugin.mount_on_user_message()
async def handle_user_message(_ctx: AgentCtx, message: ChatMessage) -> MsgSignal | None:
    """
    处理用户消息的回调函数，使用AI判断是否需要回复

    Args:
        _ctx: Agent上下文
        message: 用户消息

    Returns:
        MsgSignal | None:
            - MsgSignal.BLOCK_TRIGGER: 阻止AI响应但保存记录
            - MsgSignal.BLOCK_ALL: 完全阻止且不保存记录
            - None: 允许正常处理
    """
    try:
        core.logger.info("[AI回复过滤器] 用户消息回调被触发")

        # 获取频道类型
        channel_type = getattr(_ctx, "channel_type", None)
        channel_id = getattr(_ctx, "channel_id", "unknown")

        core.logger.info(f"[AI回复过滤器] 频道类型: {channel_type}, 频道ID: {channel_id}")

        # 检查频道类型是否启用过滤
        if channel_type == "private" and not config.ENABLE_PRIVATE:
            core.logger.info("[AI回复过滤器] 私聊过滤已禁用，直接放行")
            return None
        elif channel_type == "group" and not config.ENABLE_GROUP:
            core.logger.info("[AI回复过滤器] 群聊过滤已禁用，直接放行")
            return None
        elif channel_type not in ["private", "group"]:
            core.logger.info(f"[AI回复过滤器] 未知频道类型 ({channel_type})，直接放行")
            return None

        # 群组过滤检查
        if channel_type == "group":
            if not should_filter_group(channel_id):
                core.logger.info(f"[AI回复过滤器] 群组 {channel_id} 不在过滤范围内，直接放行")
                return None

        # 获取消息内容
        message_text = message.content_text
        core.logger.info(f"[AI回复过滤器] 分析消息: {message_text[:100]}...")

        # 获取 chat_key 用于查询历史消息
        chat_key = getattr(_ctx, "chat_key", "")

        # 🔍 调试日志：输出当前配置值
        core.logger.info(f"[AI回复过滤器] 📋 配置检查 - ENABLE_MESSAGE_MERGE: {config.ENABLE_MESSAGE_MERGE}")
        core.logger.info(f"[AI回复过滤器] 📋 配置检查 - MESSAGE_MERGE_WAIT_TIME: {config.MESSAGE_MERGE_WAIT_TIME}")
        core.logger.info(f"[AI回复过滤器] 📋 配置检查 - MESSAGE_MERGE_MAX_COUNT: {config.MESSAGE_MERGE_MAX_COUNT}")

        # 修复: 如果启用消息合并模式,直接收集消息,不立即AI判断
        if config.ENABLE_MESSAGE_MERGE:
            core.logger.info("[AI回复过滤器] 启用消息合并模式,先收集消息")
            return await handle_with_merge(_ctx, message, chat_key)

        # 未启用消息合并模式,立即调用AI判断
        should_reply = await ai_should_reply(message_text, chat_key)

        if should_reply:
            # AI判断需要回复,立即触发
            core.logger.info("[AI回复过滤器] AI判断:需要回复,返回FORCE_TRIGGER强制触发")
            return MsgSignal.FORCE_TRIGGER
        else:
            # AI判断不需要回复,根据完全接管模式决定返回信号
            if config.ENABLE_COMPLETE_TAKEOVER:
                # 完全接管模式:完全阻止,包括随机回复
                core.logger.info("[AI回复过滤器] AI判断:不需要回复,完全接管模式,返回BLOCK_ALL完全阻止")
                return MsgSignal.BLOCK_ALL
            else:
                # 非完全接管模式:阻止触发但保存记录,允许随机回复
                core.logger.info("[AI回复过滤器] AI判断:不需要回复,返回BLOCK_TRIGGER(允许随机回复)")
                return MsgSignal.BLOCK_TRIGGER

    except Exception as e:
        # 异常情况默认放行并触发，避免阻塞正常对话
        core.logger.error(f"[AI回复过滤器] 消息处理异常: {e}", exc_info=True)
        return MsgSignal.FORCE_TRIGGER



# endregion: 用户消息回调


# region: 资源清理

@plugin.mount_cleanup_method()
async def clean_up():
    """清理插件资源"""
    core.logger.info("[AI回复过滤器] 插件正在清理资源...")
    core.logger.info("[AI回复过滤器] 清理完成")


# endregion: 资源清理

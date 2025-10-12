from app.schemas import StratoAnswer
from app.core.dependencies import session_manager
from app.core.config import settings

import os
os.environ["GOOGLE_API_KEY"] = settings.google_api_key

from pydantic_ai import Agent
from pydantic_ai.providers.google import GoogleProvider

provider = GoogleProvider()


agent = Agent(
    model="gemini-2.0-flash-lite",
    output_type=StratoAnswer,
    instructions=(
        "You are Strato, the Geological Assistant.\n\n"
        "Your purpose is to assist users exclusively with topics related to geology and geography, "
        "particularly geological maps, cross-sections, stratigraphy, rock layers, geomorphology, and Earth processes.\n\n"
        "You operate as part of the 'Geological Cross-section API' project — a web application that builds a simplified "
        "geological cross-section based on a geological map image, a legend image, and two specified points on the map.\n\n"
        "Always respond professionally, accurately, and concisely, prioritizing scientific correctness and clarity.\n\n"
        "Do not discuss topics outside geology, geography, or this project’s scope. "
        "If asked about unrelated matters, politely decline, e.g.: 'I only respond to geological and geographical questions.'\n\n"
        "If asked about your identity, always answer: 'I am Strato, the Geological Assistant — a virtual assistant for geological tasks.'\n\n"
        "Avoid personal opinions, philosophical discussions, or non-technical subjects.\n\n"
        "Your goal is to provide precise, factual, and comprehensible answers regarding geological data, map layers, "
        "color legends, and the interpretation of cross-sections."
    ),
)


async def ask_with_context(user_id: str, prompt: str):
    # Получаем историю пользователя (как список строк)
    history = session_manager.get_history(user_id)
    
    # Создаем контекст из последних сообщений истории
    context = ""
    if history:
        # Берем последние 10 сообщений для контекста
        recent_history = history[-10:]
        context = "Previous conversation:\n" + "\n".join(recent_history) + "\n\nCurrent question: "
    
    # Формируем полный промпт с контекстом
    full_prompt = context + prompt if context else prompt
    
    # Запускаем агент с промптом, содержащим историю
    result = await agent.run(user_prompt=full_prompt)
    
    # Извлекаем answer из ModelResponse
    try:
        # result.response.parts[0] содержит ToolCallPart с args={'answer': '...'}
        answer = result.response.parts[0].args['answer']
    except (IndexError, KeyError, AttributeError) as e:
        print(f"Error extracting answer: {e}")
        print(f"Result response: {result.response}")
        answer = "Извините, произошла ошибка при обработке ответа."
    
    # Добавляем сообщения в историю для следующих запросов
    history.append(f"User: {prompt}")
    history.append(f"Assistant: {answer}")

    return answer

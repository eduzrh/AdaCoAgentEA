from openai import OpenAI
import concurrent.futures

client = OpenAI(
    api_key = "your_api_key",
    base_url="your_base_url",
)


def task():
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=
        [
            {"role": "user", "content": "hello"}
        ],
        stream=False
    )
    print(response)

def run_tasks(total_tasks, batch_size):
    with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
        futures = [executor.submit(task) for _ in range(total_tasks)]
        for future in concurrent.futures.as_completed(futures):
            future.result()


run_tasks(total_tasks=10, batch_size=10)
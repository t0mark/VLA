"""
NaVILA CLI Chat Node

터미널에서 내비게이션 지시를 입력하고 로봇 응답을 표시하는 대화형 CLI 노드.

Publishes:
  /navila/prompt   (std_msgs/String)  - 사용자 입력 지시z
"""

import threading

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class ChatPromptNode(Node):
    def __init__(self):
        super().__init__("chat_prompt_node")

        self.declare_parameter("prompt_topic", "/navila/prompt")

        prompt_topic = self.get_parameter("prompt_topic").value

        self.prompt_pub = self.create_publisher(String, prompt_topic, 10)

        self.input_thread = threading.Thread(target=self._stdin_reader, daemon=True)
        self.input_thread.start()

        print(
            f"\nNaVILA CLI Chat 시작\n"
            f"  prompt_topic : {prompt_topic}\n"
            f"  종료하려면 'quit' 입력\n"
        )

    def _stdin_reader(self):
        while True:
            try:
                user_input = input(">>> ")
            except EOFError:
                break

            if user_input.strip().lower() in ("quit", "exit", "q"):
                print("종료합니다.")
                rclpy.shutdown()
                break

            if not user_input.strip():
                continue

            self.prompt_pub.publish(String(data=user_input.strip()))


def main(args=None):
    rclpy.init(args=args)
    try:
        node = ChatPromptNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

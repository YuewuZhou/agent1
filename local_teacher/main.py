from agents.teacher import TeacherAgent

def main():
    # topic = "Transformers in machine learning"
    topic = "Transformers in machine learning"

    teacher = TeacherAgent()
    lesson = teacher.teach(topic)

    # paced_print(lesson, delay=2.0)
    print(lesson)

if __name__ == "__main__":
    main()

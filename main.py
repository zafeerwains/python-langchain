from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from fastapi import FastAPI
from typing import Union
import sqlite3

app = FastAPI()


import os
load_dotenv()
# llm = ChatGoogleGenerativeAI(
#     model="gemini-1.5-pro",
#     temperature=0,
#     google_api_key=os.getenv('google_api_key')
# )

# response = llm.invoke("How are you?")

# print(response)

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/items/{item_id}")
def read_item(item_id: int):
    return {"item_id": item_id}

@app.get("/students/{roll_no}")
def get_student_details(roll_no: int):
    # Connect to the database
    conn = sqlite3.connect('students.db')
    cursor = conn.cursor()

    # Query to find the student with the given roll number
    cursor.execute("SELECT * FROM students WHERE roll_no = ?", (roll_no,))
    result = cursor.fetchone()

    conn.close()

    if result:
        # Map the result to a dictionary to return
        student = {
            "roll_no": result[0],
            "name": result[1],
            "age": result[2],
            # Add other fields as necessary
        }
        return student
    else:
        return {"error": "Student not found"}

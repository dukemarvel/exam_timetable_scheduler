# Exam Timetable Scheduler

## Description
The Exam Timetable Scheduler is a web application that helps educational institutions generate exam schedules efficiently. The app allows users to upload a CSV file containing course information and select start and end dates for the exams. The application processes the input and generates an exam timetable, which can be downloaded as an Excel file.

## Features
- **CSV File Upload**: Upload a CSV file with course information.
- **Date Picker**: Select start and end dates for the exam period.
- **Schedule Generation**: Automatically generate an exam timetable.
- **Download**: Download the generated timetable as an Excel file.
- **Responsive Design**: User-friendly and responsive design.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/exam_timetable_scheduler.git
   cd exam_timetable_scheduler
   ```

2. Set up a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate   # On Windows, use `venv\Scripts\activate`
    ```
3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4. Run the application
    ```bash
    flask run or python app.py
    ```

Usage
- Upload a CSV File: Click on the "Choose File" button and select your CSV file.
- Select Dates: Use the date pickers to select the start and end dates for the exams.
- Submit: Click the "Submit" button to generate the timetable.
- Download: Once processing is complete, download the generated timetable by clicking the download link.

CSV Format
The CSV file should have the following columns:
- Course: This stands for course name and code
- Students: This stands for the population of students that registered for the course

Contributing
1. Fork the repository
2. Create a new branch (git checkout -b feature-branch-name)
3. Commit your changes (git commit -am 'Add some feature')
4. Push to the branch (git push origin feature-branch-name)
5. Create a new Pull Request

License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgments
Thank you to all contributors and open-source projects that helped in building this application.
Contact
For any questions or suggestions, please contact [josephblackduke@icloud.com].


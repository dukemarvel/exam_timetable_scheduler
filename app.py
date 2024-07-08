import pandas as pd
import numpy as np
import random
import time as tm
from datetime import datetime, timedelta
import xlsxwriter
import io
import os
import logging
from flask import Flask, request, render_template, send_file, redirect, url_for

logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)

# Set the upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load and process data
def process_data(file, start_date, end_date):
    logging.debug("Starting to process data")
    df_courses = pd.read_csv(file)
    logging.debug("Loaded CSV file")

    # Input CBT courses with fixed dates and times
    cbt_courses = {
        'Course': ['LIS 218', 'CSC 214', 'GNS 312/114', 'LIS 106', 'CSC 212', 'TCS 112', 'GSE 202', 'GNS 212', 'GNS 112', 'ICS 106', 'CSC 112'],
        'Date': ['2024-07-09', '2024-07-11', '2024-07-11', '2024-07-15', '2024-07-15', '2024-07-15', '2024-07-16', '2024-07-18', '2024-07-22', '2024-07-24', '2024-07-25'],
        'Time': ['12:00 - 1:00', '3:00 - 4:00', '8:00 - 9:00', '10:00 - 11:00', '9:00 - 10:00', '12:00 - 1:00', '8:00 - 9:00', '9:00 - 10:00', '9:00 - 10:00', '12:00 - 1:00', '8:00 - 10:00'],
        'Students': [300, 110, 1000, 533, 107, 107, 1000, 1000, 1000, 1000, 1000],
        'Location': ['CBT Centre'] * 11
    }
    
    df_cbt_courses = pd.DataFrame(cbt_courses)
    
    # Data input
    dates = pd.date_range(start=start_date, end=end_date, freq='B').strftime('%Y-%m-%d').tolist()
    time_slots = ['8:30-11:30', '12:00-3:00', '3:30-6:30']
    
    # Extract courses and students from the dataframe
    courses = df_courses['Course'].tolist()
    students = df_courses['Students'].tolist()
    
    # Combine dates and times to ensure each combination has an entry for every course
    time_combinations = [(date, time) for date in dates for time in time_slots]
    
    # Filter out Friday 12:00-3:00 PM slots
    def is_friday(date):
        return pd.to_datetime(date).weekday() == 4
    
    time_combinations = [(date, time) for date, time in time_combinations if not (is_friday(date) and time == '12:00-3:00')]
    
    # Populate the data dictionary
    data = {
        'Date': [],
        'Time': [],
        'Course': [],
        'Students': [],
        'Location': []
    }
    logging.debug("Populating data dictionary")
    for i in range(len(time_combinations)):
        date, time = time_combinations[i]
        course = courses[i % len(courses)]
        student = students[i % len(students)]
        
        data['Date'].append(date)
        data['Time'].append(time)
        data['Course'].append(course)
        data['Students'].append(student)
        data['Location'].append(None)  
    
    df = pd.DataFrame(data)
    logging.debug("Populating Concluded")
    # Append CBT courses to the dataframe
    df = pd.concat([df, df_cbt_courses], ignore_index=True)
    
    # Room capacities
    room_capacities = {
        'LR 1': 30,
        'LR 2': 30,
        'LR 3': 30,
        'LR 4': 30,
        'LR 5': 30,
        'CBT Centre': 3000,
        'CISLT': 170
    }
    
    # Assign locations based on room capacities
    def assign_locations(row):
        if row['Location'] == 'CBT Centre':
            return 'CBT Centre'
        
        total_students = row['Students']
        locations = []
        for room, capacity in room_capacities.items():
            if total_students <= 0:
                break
            if capacity <= total_students:
                locations.append(f"{room}({capacity})")
                total_students -= capacity
            else:
                locations.append(f"{room}({total_students})")
                total_students = 0
        return ', '.join(locations)
    
    df['Location'] = df.apply(assign_locations, axis=1)

    # Combine Course and Location for the final display
    df['Course_Location'] = df['Course'] + ' (' + df['Students'].astype(str) + ')\n' + df['Location']

    # Separate CBT courses from the rest of the courses for shuffling
    cbt_rows = df[df['Location'] == 'CBT Centre']
    non_cbt_rows = df[df['Location'] != 'CBT Centre']
    
    # Generate initial population
    def initialize_population(size):
        logging.debug("Initializing population")
        population = []
        for i in range(size):
            logging.debug(f"Initializing individual {i+1} of {size}")
            shuffled_non_cbt = non_cbt_rows.sample(frac=1).reset_index(drop=True)
            individual = pd.concat([shuffled_non_cbt, cbt_rows]).reset_index(drop=True)
            population.append(individual)
        logging.debug("Population initialized")
        return population
        
    # Fitness function
    def fitness(individual):
        conflicts = 0
        for i, exam1 in individual.iterrows():
            for j, exam2 in individual.iterrows():
                if i != j and exam1['Time'] == exam2['Time'] and exam1['Date'] == exam2['Date']:
                    locations1 = exam1['Location'].split(', ')
                    locations2 = exam2['Location'].split(', ')
                    for loc1 in locations1:
                        for loc2 in locations2:
                            if loc1 == loc2:
                                conflicts += 1
            # Check room capacities
            total_students = exam1['Students']
            assigned_locations = []
            for loc in room_capacities:
                if total_students <= 0:
                    break
                if loc in exam1['Location']:
                    total_students -= room_capacities[loc]
                    assigned_locations.append(loc)
            if total_students > 0:
                conflicts += total_students  # Not all students can be accommodated
        return -conflicts  # Fewer conflicts is better
    
    # Selection
    def selection(population):
        population.sort(key=fitness, reverse=True)
        return population[:len(population)//2]
    
    # Crossover
    def crossover(parent1, parent2):
        crossover_point = len(parent1) // 2
        child1 = pd.concat([parent1.iloc[:crossover_point], parent2.iloc[crossover_point:]]).reset_index(drop=True)
        child2 = pd.concat([parent2.iloc[:crossover_point], parent1.iloc[crossover_point:]]).reset_index(drop=True)
        return child1, child2
    
    # Mutation
    def mutate(individual):
        idx = random.randint(0, len(non_cbt_rows)-1)  # Ensure mutation does not affect CBT courses
        individual.loc[idx] = non_cbt_rows.sample().iloc[0]
    
    # Genetic Algorithm
    def genetic_algorithm(df, population_size, generations):
        start_time = tm.time()
        population = initialize_population(population_size)
        
        for generation in range(generations):
            logging.debug("generations started")
            population = selection(population)
            new_population = []
            while len(new_population) < population_size:
                parent1, parent2 = random.sample(population, 2)
                child1, child2 = crossover(parent1, parent2)
                if random.random() < 0.1:
                    mutate(child1)
                if random.random() < 0.1:
                    mutate(child2)
                new_population.extend([child1, child2])
            population = new_population
        logging.debug("generations ended")
        best_solution = max(population, key=fitness)
        logging.debug("best_solution gotten")
        end_time = tm.time()
        simulation_time = end_time - start_time
        logging.debug("simulation_time calculated")
        convergence_rate = fitness(best_solution)
        logging.debug("convergence_rate done")

        
        return best_solution, simulation_time, convergence_rate

    # Simulated Annealing
    def simulated_annealing(initial_solution, temp, cooling_rate):
        start_time = tm.time()
        current_solution = initial_solution
        current_fitness = fitness(current_solution)
        logging.debug("SA Started")
        while temp > 1:
            neighbor = current_solution.copy()
            mutate(neighbor)
            neighbor_fitness = fitness(neighbor)
            if neighbor_fitness > current_fitness or random.uniform(0, 1) < np.exp((neighbor_fitness - current_fitness) / temp):
                current_solution = neighbor
                current_fitness = neighbor_fitness
            temp *= cooling_rate
        end_time = tm.time()
        simulation_time = end_time - start_time
        convergence_rate = current_fitness
        logging.debug("SA Concluded")
        return current_solution, simulation_time, convergence_rate

    return df, genetic_algorithm, simulated_annealing

# Routes
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        start_date = request.form['start_date']
        end_date = request.form['end_date']
        
        df, genetic_algorithm, simulated_annealing = process_data(file, start_date, end_date)
        logging.debug("File processed")
        
        # Genetic Algorithm parameters
        population_size = 10
        generations = 50
        
        # Run Genetic Algorithm
        best_solution_ga, ga_simulation_time, ga_convergence_rate = genetic_algorithm(df, population_size, generations)
        
        # Simulated Annealing parameters
        initial_temperature = 100
        cooling_rate = 0.99
        
        
        # Run Simulated Annealing
        best_solution_sa, sa_simulation_time, sa_convergence_rate = simulated_annealing(df, initial_temperature, cooling_rate)

        # Format the final solution for display
        def format_schedule(df):
            df['Combined'] = df['Course_Location']
            schedule = pd.pivot_table(df, values='Combined', index='Date', columns='Time', aggfunc=lambda x: ' '.join(str(v) for v in x))
            return schedule

        # Format schedules
        ga_formatted_schedule = format_schedule(best_solution_ga)
        sa_formatted_schedule = format_schedule(best_solution_sa)

        # Metrics
        metrics = {
            'Algorithm': ['Genetic Algorithm', 'Simulated Annealing'],
            'Simulation Time (s)': [ga_simulation_time, sa_simulation_time],
            'Convergence Rate': [ga_convergence_rate, sa_convergence_rate]
        }
        metrics_df = pd.DataFrame(metrics)

        # Save the formatted schedules to an Excel file
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            best_solution_ga.to_excel(writer, sheet_name='GA Timetable', index=False)
            best_solution_sa.to_excel(writer, sheet_name='SA Timetable', index=False)
            
            ga_formatted_schedule.to_excel(writer, sheet_name='GA Formatted Timetable')
            sa_formatted_schedule.to_excel(writer, sheet_name='SA Formatted Timetable')
            
            metrics_df.to_excel(writer, sheet_name='Metrics', index=False)
            
            # Formatting
            worksheet_ga = writer.sheets['GA Timetable']
            worksheet_sa = writer.sheets['SA Timetable']
            worksheet_ga_formatted = writer.sheets['GA Formatted Timetable']
            worksheet_sa_formatted = writer.sheets['SA Formatted Timetable']
            worksheet_metrics = writer.sheets['Metrics']
            
            worksheet_ga.set_column('A:D', 20)
            worksheet_sa.set_column('A:D', 20)
            worksheet_ga_formatted.set_column('A:D', 30)
            worksheet_sa_formatted.set_column('A:D', 30)
            worksheet_metrics.set_column('A:C', 25)

        output.seek(0)
        return send_file(output, download_name='exam_schedule.xlsx', as_attachment=True)

        
@app.route('/download/<filename>')
def download_file(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))

if __name__ == '__main__':
    app.run(debug=True)

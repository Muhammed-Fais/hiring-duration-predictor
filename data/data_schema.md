# Data Schema for Job Requisition Time-to-Fill Prediction

This document outlines the data fields to be collected for each job requisition. Accurate and comprehensive data is crucial for building an effective forecasting model.

## Requisition Details

| Field Name                 | Data Type     | Description                                                                                                | Example                                  | Importance |
|----------------------------|---------------|------------------------------------------------------------------------------------------------------------|------------------------------------------|------------|
| `RequisitionID`            | String/Integer| Unique identifier for the job requisition.                                                                 | "REQ2023-001", 10345                     | High       |
| `JobTitle`                 | String        | The official title of the position.                                                                        | "Software Engineer", "Marketing Manager" | High       |
| `Department`               | String        | The department the role belongs to.                                                                        | "Technology", "Sales", "Human Resources" | High       |
| `JobLevel`                 | String        | Seniority level of the role. Standardize this (e.g., Intern, Junior, Mid-Level, Senior, Lead, Manager, Director, VP, C-Level). | "Senior", "Mid-Level"                | High       |
| `LocationCity`             | String        | City where the job is located. Use "Remote" if applicable.                                                 | "New York", "London", "Remote"           | High       |
| `LocationCountry`          | String        | Country where the job is located. Use "Remote" if applicable.                                              | "USA", "UK", "Remote"                    | High       |
| `EmploymentType`           | String        | Type of employment (e.g., Full-time, Part-time, Contract).                                                 | "Full-time"                              | Medium     |
| `IsRemote`                 | Boolean       | True if the position is fully remote, False otherwise.                                                     | True, False                              | High       |
| `IsHybrid`                 | Boolean       | True if the position is hybrid, False otherwise.                                                           | True, False                              | Medium     |
| `DateRequisitionOpened`    | Date          | The date the job requisition was officially opened. (YYYY-MM-DD)                                           | "2023-01-15"                             | High       |
| `DateRequisitionClosed`    | Date          | The date the job requisition was successfully filled or closed. (YYYY-MM-DD) (Target variable derivation)  | "2023-03-10"                             | High       |
| `HiringManagerID`          | String/Integer| Unique identifier for the hiring manager.                                                                  | "HMGR-056"                               | Medium     |
| `RecruiterID`              | String/Integer| Unique identifier for the primary recruiter assigned.                                                      | "REC-007"                                | Medium     |
| `NumberOfOpenings`         | Integer       | Number of positions to be filled for this requisition.                                                     | 1, 5                                     | Medium     |
| `SalaryRangeMin`           | Float         | Minimum of the salary range offered (optional, if available and consistent).                               | 60000                                    | Low        |
| `SalaryRangeMax`           | Float         | Maximum of the salary range offered (optional).                                                            | 90000                                    | Low        |
| `PriorityLevel`            | String        | Urgency of the requisition (e.g., High, Medium, Low).                                                      | "High"                                   | Medium     |
| `RequiredYearsExperienceMin`| Integer       | Minimum years of experience required.                                                                      | 3                                        | High       |
| `RequiredYearsExperienceMax`| Integer       | Maximum years of experience suggested (can be same as Min for specific targets).                           | 5                                        | Medium     |

## Skills and Qualifications

| Field Name                 | Data Type     | Description                                                                                                | Example                                  | Importance |
|----------------------------|---------------|------------------------------------------------------------------------------------------------------------|------------------------------------------|------------|
| `PrimarySkills`            | List[String]  | Comma-separated list or JSON array of essential skills. Standardize skill names.                         | "Python, SQL, Machine Learning"          | High       |
| `SecondarySkills`          | List[String]  | Comma-separated list or JSON array of desirable skills.                                                    | "AWS, Docker, Agile"                     | Medium     |
| `EducationLevelRequired`   | String        | Minimum education level required (e.g., Bachelor's, Master's, PhD, None).                                  | "Bachelor's"                             | Medium     |

## (Potentially) Company/Contextual Data (Collected at time of opening)

| Field Name                       | Data Type     | Description                                                                                                   | Example      | Importance |
|----------------------------------|---------------|---------------------------------------------------------------------------------------------------------------|--------------|------------|
| `TotalOpenRequisitionsCompany`   | Integer       | Total number of open requisitions in the company at the time this requisition is opened.                      | 75           | Medium     |
| `TotalOpenRequisitionsDept`      | Integer       | Total number of open requisitions in the same department at the time this requisition is opened.              | 10           | Medium     |
| `RecruiterWorkload`              | Integer       | Number of open requisitions currently assigned to `RecruiterID` when this requisition is opened.              | 15           | Medium     |
| `QuarterOpened`                  | String/Integer| The fiscal or calendar quarter the requisition was opened in (e.g., "Q1", "Q2-2023").                         | "Q1-2023"    | Medium     |

## Target Variable (to be calculated)

| Field Name         | Data Type | Description                                       | Calculation                            |
|--------------------|-----------|---------------------------------------------------|----------------------------------------|
| `TimeToFill_days`  | Integer   | Number of days from opening to closing.           | `DateRequisitionClosed` - `DateRequisitionOpened` |

**Notes:**

*   **Consistency is Key:** Ensure data is entered consistently (e.g., for `JobLevel`, `Department`, `PrimarySkills`). Consider using dropdowns or predefined lists in your data entry system.
*   **Historical Data:** Collect as much historical data as possible. The more data, the better the model.
*   **Job Description Text:** While not listed as a structured field here, the full job description text can be very valuable if you plan to use Natural Language Processing (NLP) techniques for feature engineering. This would be an advanced step.
*   This schema is a starting point. It can be refined based on data availability and initial analysis. 
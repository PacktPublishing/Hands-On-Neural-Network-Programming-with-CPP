#include <iostream>
#include <fstream>
#include <exception>
#include <string>
#include <algorithm>

#include "csv.h"

std::vector<std::vector<std::string>> loadDataset(const std::string &filepath)
{
    std::vector<std::vector<std::string>> result(36, std::vector<std::string>(858 + 1));
    io::CSVReader<36> csvReader(filepath);
    csvReader.set_header("Age", "Number of sexual partners", "First sexual intercourse", "Num of pregnancies", 
        "Smokes", "Smokes (years)", "Smokes (packs/year)", "Hormonal Contraceptives", "Hormonal Contraceptives (years)", "IUD", "IUD (years)", 
        "STDs", "STDs (number)", "STDs:condylomatosis", "STDs:cervical condylomatosis", "STDs:vaginal condylomatosis", 
        "STDs:vulvo-perineal condylomatosis", "STDs:syphilis", "STDs:pelvic inflammatory disease", "STDs:genital herpes", "STDs:molluscum contagiosum", 
        "STDs:AIDS", "STDs:HIV", "STDs:Hepatitis B", "STDs:HPV", "STDs: Number of diagnosis", "STDs: Time since first diagnosis", 
        "STDs: Time since last diagnosis", "Dx:Cancer", "Dx:CIN", "Dx:HPV", "Dx", "Hinselmann", "Schiller", "Citology", "Biopsy");
    std::string Age, Number_of_sexual_partners, First_sexual_intercourse, Num_of_pregnancies, Smokes, Smokes_years, 
        Smokes_packs_year, Hormonal_Contraceptives, Hormonal_Contraceptives_years, IUD, IUD_years, STDs, STDs_number, STDs_condylomatosis, 
        STDs_cervical_condylomatosis, STDs_vaginal_condylomatosis, STDs_vulvo_perineal_condylomatosis, STDs_syphilis, STDs_pelvic_inflammatory_disease, 
        STDs_genital_herpes, STDs_molluscum_contagiosum, STDs_AIDS, STDs_HIV, STDs_Hepatitis_B, STDs_HPV, STDs_Number_of_diagnosis, STDs_Time_since_first_diagnosis, 
        STDs_Time_since_last_diagnosis, Dx_Cancer, Dx_CIN, Dx_HPV, Dx, Hinselmann, Schiller, Citology, Biopsy;
    
    int rowIndex = 0;
    while (csvReader.read_row(Age, Number_of_sexual_partners, First_sexual_intercourse, Num_of_pregnancies, Smokes, Smokes_years, 
    Smokes_packs_year, Hormonal_Contraceptives, Hormonal_Contraceptives_years, IUD, IUD_years, STDs, STDs_number, STDs_condylomatosis, 
    STDs_cervical_condylomatosis, STDs_vaginal_condylomatosis, STDs_vulvo_perineal_condylomatosis, STDs_syphilis, STDs_pelvic_inflammatory_disease, 
    STDs_genital_herpes, STDs_molluscum_contagiosum, STDs_AIDS, STDs_HIV, STDs_Hepatitis_B, STDs_HPV, STDs_Number_of_diagnosis, STDs_Time_since_first_diagnosis, 
    STDs_Time_since_last_diagnosis, Dx_Cancer, Dx_CIN, Dx_HPV, Dx, Hinselmann, Schiller, Citology, Biopsy))
    {
        std::vector<std::string> row{Age, Number_of_sexual_partners, First_sexual_intercourse, Num_of_pregnancies, Smokes, Smokes_years, 
            Smokes_packs_year, Hormonal_Contraceptives, Hormonal_Contraceptives_years, IUD, IUD_years, STDs, STDs_number, STDs_condylomatosis, 
            STDs_cervical_condylomatosis, STDs_vaginal_condylomatosis, STDs_vulvo_perineal_condylomatosis, STDs_syphilis, STDs_pelvic_inflammatory_disease, 
            STDs_genital_herpes, STDs_molluscum_contagiosum, STDs_AIDS, STDs_HIV, STDs_Hepatitis_B, STDs_HPV, STDs_Number_of_diagnosis, STDs_Time_since_first_diagnosis, 
            STDs_Time_since_last_diagnosis, Dx_Cancer, Dx_CIN, Dx_HPV, Dx, Hinselmann, Schiller, Citology, Biopsy};

        for(unsigned col = 0; col < row.size(); ++col)
        {
            result[col][rowIndex] = row[col];
        }
        rowIndex++;
    }
    return result;
}

int main(int, char **)
{
    try
    {
        auto cervicalCancerDS = loadDataset("../data/risk_factors_cervical_cancer.csv");
        std::vector<std::tuple<int, std::string>> list;   list.reserve(36);
        for(auto column : cervicalCancerDS) {
            int numberOfMissingData = std::count(column.begin(), column.end(), "?");
            list.push_back(std::make_tuple(numberOfMissingData, column[0]));
        }
        std::sort(list.begin(), list.end(), [](const std::tuple<int, std::string> &a, const std::tuple<int, std::string> &b) { 
                return std::get<0>(a) > std::get<0>(b); 
            });

        std::cout << std::fixed;    std::cout << std::setprecision(1);
        for(auto item : list) {
            int nas = std::get<0>(item);
            std::cout << nas << " (" << nas/8.58 << "%)\t|\t" << std::get<1>(item) << "\n";
        }
    }
    catch (std::exception const &e)
    {
        std::cerr << "Exception: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
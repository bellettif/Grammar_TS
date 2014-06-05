#ifndef DATA_H
#define DATA_H

#include <string>

#include "string_utils.h"

#include "file_reader.h"


static const std::string folder_path = "/Users/francois/Grammar_TS/data/";

static const std::string achu_1 = file_reader::translate_to_chars(
            string_utils::replace_copy(
            file_reader::read_lines_from_file(folder_path + "achuSeq_1.csv").front(),
            ",",
             " ")
        );
static const std::string achu_2 = file_reader::translate_to_chars(
            string_utils::replace_copy(
            file_reader::read_lines_from_file(folder_path + "achuSeq_2.csv").front(),
            ",",
             " ")
        );
static const std::string achu_3 = file_reader::translate_to_chars(
            string_utils::replace_copy(
            file_reader::read_lines_from_file(folder_path + "achuSeq_3.csv").front(),
            ",",
            " ")
        );
static const std::string achu_4 = file_reader::translate_to_chars(
            string_utils::replace_copy(
            file_reader::read_lines_from_file(folder_path + "achuSeq_4.csv").front(),
            ",",
            " ")
        );
static const std::string achu_5 = file_reader::translate_to_chars(
            string_utils::replace_copy(
            file_reader::read_lines_from_file(folder_path + "achuSeq_5.csv").front(),
            ",",
            " ")
        );
static const std::string achu_6 = file_reader::translate_to_chars(
            string_utils::replace_copy(
            file_reader::read_lines_from_file(folder_path + "achuSeq_6.csv").front(),
            ",",
            " ")
        );
static const std::string achu_7 = file_reader::translate_to_chars(
            string_utils::replace_copy(
            file_reader::read_lines_from_file(folder_path + "achuSeq_7.csv").front(),
            ",",
            " ")
        );
static const std::string achu_8 = file_reader::translate_to_chars(
            string_utils::replace_copy(
            file_reader::read_lines_from_file(folder_path + "achuSeq_8.csv").front(),
            ",",
            " ")
        );
static const std::string achu_9 = file_reader::translate_to_chars(
            string_utils::replace_copy(
            file_reader::read_lines_from_file(folder_path + "achuSeq_9.csv").front(),
            ",",
            " ")
        );
static const std::string oldo_1 = file_reader::translate_to_chars(
            string_utils::replace_copy(
            file_reader::read_lines_from_file(folder_path + "oldoSeq_1.csv").front(),
            ",",
            " ")
        );
static const std::string oldo_2 = file_reader::translate_to_chars(
            string_utils::replace_copy(
            file_reader::read_lines_from_file(folder_path + "oldoSeq_2.csv").front(),
            ",",
            " ")
        );
static const std::string oldo_3 = file_reader::translate_to_chars(
            string_utils::replace_copy(
            file_reader::read_lines_from_file(folder_path + "oldoSeq_3.csv").front(),
            ",",
            " ")
        );
static const std::string oldo_4 = file_reader::translate_to_chars(
            string_utils::replace_copy(
            file_reader::read_lines_from_file(folder_path + "oldoSeq_4.csv").front(),
            ",",
            " ")
        );
static const std::string oldo_5 = file_reader::translate_to_chars(
            string_utils::replace_copy(
            file_reader::read_lines_from_file(folder_path + "oldoSeq_5.csv").front(),
            ",",
            " ")
        );
static const std::string oldo_6 = file_reader::translate_to_chars(
            string_utils::replace_copy(
            file_reader::read_lines_from_file(folder_path + "oldoSeq_6.csv").front(),
            ",",
            " ")
        );
static const std::string oldo_7 = file_reader::translate_to_chars(
            string_utils::replace_copy(
            file_reader::read_lines_from_file(folder_path + "oldoSeq_8.csv").front(),
            ",",
            " ")
        );
static const std::string oldo_8 = file_reader::translate_to_chars(
            string_utils::replace_copy(
            file_reader::read_lines_from_file(folder_path + "oldoSeq_9.csv").front(),
            ",",
            " ")
        );
static const std::string oldo_9 = file_reader::translate_to_chars(
            string_utils::replace_copy(
            file_reader::read_lines_from_file(folder_path + "oldoSeq_10.csv").front(),
            ",",
            " ")
        );



#endif // DATA_H

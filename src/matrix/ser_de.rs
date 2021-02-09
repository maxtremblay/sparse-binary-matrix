use super::SparseBinMat;
use serde;
use serde::de::{Deserializer, MapAccess, Visitor};
use serde::ser::{Serialize, SerializeStruct, Serializer};
use serde::Deserialize;
use std::fmt;

impl Serialize for SparseBinMat {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let rows: Vec<_> = self
            .rows()
            .map(|row| row.to_owned().to_positions_vec())
            .collect();
        let mut state = serializer.serialize_struct("SparseBinMat", 2)?;
        state.serialize_field("number_of_columns", &self.number_of_columns())?;
        state.serialize_field("rows", &rows)?;
        state.end()
    }
}

#[derive(Deserialize)]
#[serde(field_identifier, rename_all = "snake_case")]
enum Field {
    NumberOfColumns,
    Rows,
}

const FIELDS: &'static [&'static str] = &["number_of_columns", "rows"];

struct MatrixVisitor;

impl<'de> Visitor<'de> for MatrixVisitor {
    type Value = SparseBinMat;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("struct SparseBinMat")
    }

    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut number_of_columns = None;
        let mut rows = None;
        while let Some(key) = map.next_key()? {
            match key {
                Field::NumberOfColumns => {
                    if number_of_columns.is_some() {
                        return Err(serde::de::Error::duplicate_field("number_of_columns"));
                    }
                    number_of_columns = Some(map.next_value()?);
                }
                Field::Rows => {
                    if rows.is_some() {
                        return Err(serde::de::Error::duplicate_field("rows"));
                    }
                    rows = Some(map.next_value()?);
                }
            }
        }
        let number_of_columns: usize = number_of_columns
            .ok_or_else(|| serde::de::Error::missing_field("number_of_columns"))?;
        let rows: Vec<Vec<usize>> = rows.ok_or_else(|| serde::de::Error::missing_field("rows"))?;
        if number_of_columns == 0 && rows.len() == 0 {
            Ok(SparseBinMat::empty())
        } else {
            SparseBinMat::try_new(number_of_columns, rows)
                .map_err(|error| serde::de::Error::custom(&error.to_string()))
        }
    }
}

impl<'de> Deserialize<'de> for SparseBinMat {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_struct("SparseBinMat", FIELDS, MatrixVisitor)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use serde_test::{assert_de_tokens_error, assert_tokens, Token};

    #[test]
    fn ser_de_empty_matrix() {
        let matrix = SparseBinMat::empty();
        assert_tokens(
            &matrix,
            &[
                Token::Struct {
                    name: "SparseBinMat",
                    len: 2,
                },
                Token::String(&"number_of_columns"),
                Token::U64(0),
                Token::String(&"rows"),
                Token::Seq { len: Some(0) },
                Token::SeqEnd,
                Token::StructEnd,
            ],
        );
    }

    #[test]
    fn ser_de_2_by_5_matrix() {
        let matrix = SparseBinMat::new(5, vec![vec![0, 2, 4], vec![1, 3]]);
        assert_tokens(
            &matrix,
            &[
                Token::Struct {
                    name: "SparseBinMat",
                    len: 2,
                },
                Token::String(&"number_of_columns"),
                Token::U64(5),
                Token::String(&"rows"),
                Token::Seq { len: Some(2) },
                Token::Seq { len: Some(3) },
                Token::U64(0),
                Token::U64(2),
                Token::U64(4),
                Token::SeqEnd,
                Token::Seq { len: Some(2) },
                Token::U64(1),
                Token::U64(3),
                Token::SeqEnd,
                Token::SeqEnd,
                Token::StructEnd,
            ],
        );
    }

    #[test]
    fn de_unsorted_rows() {
        assert_de_tokens_error::<SparseBinMat>(
            &[
                Token::Struct {
                    name: "SparseBinMat",
                    len: 2,
                },
                Token::String(&"number_of_columns"),
                Token::U64(5),
                Token::String(&"rows"),
                Token::Seq { len: Some(1) },
                Token::Seq { len: Some(3) },
                Token::U64(0),
                Token::U64(4),
                Token::U64(2),
                Token::SeqEnd,
                Token::SeqEnd,
                Token::StructEnd,
            ],
            "some positions are not sorted",
        );
    }
}

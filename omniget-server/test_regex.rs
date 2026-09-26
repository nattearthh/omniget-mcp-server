use regex::Regex;

fn main() {
    let re = Regex::new(r#"https:(?:\\?/){2}scontent[^"'\s<>]+\.(?:jpg|png|webp)[^"'\s<>]*"#).unwrap();
    let text = r#"{"image": "https:\/\/scontent.bkk1-1.fna.fbcdn.net\/v\/t39.30808-6\/461146241_10114026953826931_792505566276859556_n.jpg?_nc_cat=105&ccb=1-7&_nc_sid=a5f9e3&_nc_eui2=AeFwT2V3w9-Q"} 
    and a normal one: https://scontent.bkk1-1.fna.fbcdn.net/v/t39.30808-6/461146241_10114026953826931_792505566276859556_n.jpg?_nc_cat=105"#;
    
    for mat in re.find_iter(text) {
        println!("Match: {}", mat.as_str());
    }
}

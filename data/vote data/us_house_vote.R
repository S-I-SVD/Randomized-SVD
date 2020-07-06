# set wd
setwd("~/GitHub/CompLegFall2019/data/us_upper_congress/Vote")

# remove objects
rm(list=ls())
# detach all libraries
detachAllPackages <- function() {
  basic.packages <- c("package:stats","package:graphics","package:grDevices","package:utils","package:datasets","package:methods","package:base")
  package.list <- search()[ifelse(unlist(gregexpr("package:",search()))==1,TRUE,FALSE)]
  package.list <- setdiff(package.list,basic.packages)
  if (length(package.list)>0)  for (package in package.list) detach(package, character.only=TRUE)
}
detachAllPackages()

# load libraries
pkgTest <- function(pkg){
  new.pkg <- pkg[!(pkg %in% installed.packages()[, "Package"])]
  if (length(new.pkg))
    install.packages(new.pkg, dependencies = TRUE)
  sapply(pkg, require, character.only = TRUE)
}

lapply(c("stringr", "dplyr", "plyr", "tidyverse", "rvest", "zoo", "lubridate", "XML"), pkgTest)

# create an empty data frame for data storage
ind_year_list <- list()

# Collect votes
for (year in 1990:2020) {
  # max amount of vote in each year
  max <- read_html(paste0("http://clerk.house.gov/evs/",year,"/index.asp")) %>%
    html_node("tr:nth-child(2) td > a") %>%
    html_text() %>%
    as.numeric()
  
  ind_roll_list <- list()
  
  for (roll_number in 1:max) {
    # xml file
    if(roll_number<10){
      link <- paste0("http://clerk.house.gov/evs/",year,"/roll00",roll_number,".xml")
    } else if(roll_number<100){
      link <- paste0("http://clerk.house.gov/evs/",year,"/roll0",roll_number,".xml")
    } else{
      link <- paste0("http://clerk.house.gov/evs/",year,"/roll",roll_number,".xml")
    }
    # parse xml file
    genlist <- xmlToList(xmlParse(link))
    
    
    # collect general data
    majority <- genlist[[1]]$majority # majority party
    congress <- genlist[[1]]$congress # number of congress
    session <- genlist[[1]]$session # session of congress, 1st or 2nd
    chamber <- 1 # change to 2 for senate, 1 is for house of rep
    # roll_number is the index of roll call vote within each year
    bill <- genlist[[1]]$`legis-num` # bill number, i.e. S 3546
      if(is.null(bill)){ bill = " "}
    bill_title <- genlist[[1]]$`vote-desc` #only available for years after 2003
      if(is.null(bill_title)) {bill_title=" "}
    vote_question <- genlist[[1]]$`vote-question`
      if(is.null(vote_question)) {vote_question=" "}
    vote_type <- genlist[[1]]$`vote-type`
    vote_result <- genlist[[1]]$`vote-result`
    
    if(is.null(vote_result)){next}
    
    date <- as.Date(genlist[[1]]$`action-date`,format="%d-%b-%Y")
    
    # collect inddividual vote data
    ind_vote_list <- list()
    for (i in 1:(lengths(genlist)[2])) {
      
      if(year<2003){
        last_name <- genlist[[2]][[i]]$legislator$text
        party <- genlist[[2]][[i]]$legislator$.attrs[1]
        state <- genlist[[2]][[i]]$legislator$.attrs[2]
        role <- genlist[[2]][[i]]$legislator$.attrs[3]
      } else{
        last_name <- genlist[[2]][[i]]$legislator$text
        party <- genlist[[2]][[i]]$legislator$.attrs[4]
        state <- genlist[[2]][[i]]$legislator$.attrs[5]
        role <- genlist[[2]][[i]]$legislator$.attrs[6]
      }
      
      vote <- genlist[[2]][[i]]$vote
      
      ind_vote <- data.frame(year, roll_number, congress, session, chamber, 
                             bill, vote_question, vote_type, date,
                             last_name, party, state, role, vote,
                             majority, vote_result, stringsAsFactors = F)
      ind_vote_list[[i]] <- ind_vote
    }
    
    # data frame of votes within each bill
    ind_roll <- do.call(rbind, ind_vote_list)
    ind_roll_list[[i]] <- ind_roll
  }
  # data frame of votes within each year
  ind_year <- do.call(rbind,ind_roll_list)
  # ind_year_list[[i]] <- ind_year
  
  
  # export data
  write.csv(ind_year, paste0("us_house_vote_",year,".csv"))
  
}
# # all votes since 1970
# gen_vote <- do.call(rbind,ind_year_list)

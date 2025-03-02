

with temp_temp as (
    select *, 
    date_format( transaction_date, '%Y-%m') as _months
    from transactions 

), 
sum_tabels as (
    select user_id, 
        _months, 
    sum(amount)  as monthly_spending
    from temp_temp
    group by user_id, _months  
)
select id, 
    user_id, 
    amount, 
    transaction_date

from (
    select * , 
    row_number() over( partition by _months order by monthly_spending desc ) as spending_ranks
    from sum_tabels
)
where spending_ranks <= 3 ; 

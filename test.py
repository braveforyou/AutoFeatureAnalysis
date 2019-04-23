x1=['xinyanmaxoverduedays', 'xinyancurrentconsfinproductcount', 'xinyanbehaviorloansorgcount', 'xinyancurrentloansproductcount', 'tdidnumberhujin30dayscrossplatform', 'tdidnumberloan7dayscrossplatform', 'overviewlastcalltime', 'xinyancurrentloansorgcount', 'last30dayscalltotaltimes', 'lasttwoweekscalldurationbetween15and30', 'lasttwoweekscallintimes', 'lasttwoweekscalltotaltimes', 'overviewfirstcalltime', 'lasttwoweekscallteltotalnums', 'shumei_itfin_loan_approval_level', 'shumei_itfin_loan_overdue_level', 'shumei_debit_salary_level', 'shumei_credit_loan_overdues', 'shumei_credit_loan_overdue_level', 'shumei_credit_loan_approval_level', 'tdidnumberloan3monthscrossplatform', 'lasttwoweekscalldurationbelow15', 'tdmobile30dayscrossplatform', 'baiduidphonequeryorgcntm6', 'tdmobileloan7dayscrossplatform', 'tdidnumberhujin7dayscrossplatform', 'xinyancurrentconsfincredibility', 'caller_call_people_24_2_6', 'baiduidphonequerytimesd30', 'xinyanapplylatestthreemonth', 'tdmobile3monthscrossplatform', 'xinyanbehaviorlatestonemonthfail', 'tdmobileloan3monthscrossplatform', 'sum_financecorp_count']


x2=['idcard_address', 'xinyanmaxoverduedays', 'xinyancurrentconsfinproductcount', 'xinyanbehaviorloansorgcount',
'xinyancurrentloansproductcount', 'tdidnumberhujin30dayscrossplatform', 'tdidnumberloan7dayscrossplatform',
'overviewlastcalltime', 'xinyancurrentloansorgcount', 'last30dayscalltotaltimes', 'lasttwoweekscalldurationbetween15and30',
'lasttwoweekscallintimes', 'lasttwoweekscalltotaltimes', 'overviewfirstcalltime', 'lasttwoweekscallteltotalnums',
 'shumei_itfin_loan_approval_level', 'shumei_itfin_loan_overdue_level', 'shumei_debit_salary_level',
 'shumei_credit_loan_overdues', 'shumei_credit_loan_overdue_level', 'shumei_credit_loan_approval_level',
 'tdidnumberloan3monthscrossplatform', 'lasttwoweekscalldurationbelow15', 'tdmobile30dayscrossplatform',
 'baiduidphonequeryorgcntm6', 'tdmobileloan7dayscrossplatform', 'tdidnumberhujin7dayscrossplatform',
 'xinyancurrentconsfincredibility', 'caller_call_people_24_2_6', 'baiduidphonequerytimesd30',
 'xinyanapplylatestthreemonth', 'tdmobile3monthscrossplatform', 'xinyanbehaviorlatestonemonthfail',
 'tdmobileloan3monthscrossplatform', 'sum_financecorp_count']


result=[i for i in x2 if i not in x1]

print(result)
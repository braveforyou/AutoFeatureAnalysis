#!/usr/bin/python
# coding:utf-8
attributeList = [
    'id_no_de',
    'mobile_de',
    'sex',
    'age',
    'marital_status',
    'home_province',
    'home_city',
    'sum_financecorp_count',
    'max_creditlimit_amount',
    'car_no',
    'mail',
    'face_value',
    'idcard_address',
    'idcard_valid_period',
    'status',
    'last_loan_date', # todo 需要处理，上一次借款时间需要处理成距离上一次借款多久了
    'idbaiduloan2manyscore',
    'baiduidquerytimesd7',
    'baiduidphonequerytimesd7',
    'baiduidquerytimesd15',
    'baiduidphonequerytimesd15',
    'baiduidquerytimesd30',
    'baiduidphonequerytimesd30',
    'baiduidquerytimesm3',
    'baiduidphonequerytimesm3',
    'baiduidquerytimesm6',
    'baiduidphonequerytimesm6',
    'baiduidquerytimesm12',
    'baiduidphonequerytimesm12',
    'baiduidquerytimesy1',
    'baiduidphonequerytimesy1',
    'baiduidqueryorgcntd7',
    'baiduidphonequeryorgcntd7',
    'baiduidqueryorgcntd15',
    'baiduidphonequeryorgcntd15',
    'baiduidqueryorgcntd30',
    'baiduidphonequeryorgcntd30',
    'baiduidqueryorgcntm3',
    'baiduidphonequeryorgcntm3',
    'baiduidqueryorgcntm6',
    'baiduidphonequeryorgcntm6',
    'baiduidqueryorgcntm12',
    'baiduidphonequeryorgcntm12',
    'baiduidqueryorgcnty1',
    'baiduidphonequeryorgcnty1',
    'blacklistnamewithphone',
    'blacklistnamewithidcard',
    'weightbeblack',
    'weightblack',
    'weighttoblack',
    'cntbeblack',
    'cntblack',
    'cntblack2',
    'cntblackdivcntall',
    'cntrouter',
    'cnttoblack',
    'router_ratio',
    'callcntbeblack',
    'callcnttoblack',
    'timespentbeblack',
    'timespenttoblack',
    'phonegrayscore',
    'huaceappstability7d',
    'huaceappstability90d',
    'huaceappstability180d',
    'huacecar7d',
    'huacecar90d',
    'huacecar180d',
    'huacedevicebrand',
    'huacedeviceprice',
    'huacedevicerank',
    'huacefinance7d',
    'huacefinance90d',
    'huacefinance180d',
    'huacelaunchday',
    'huaceloan7d',
    'huaceloan90d',
    'huaceloan180d',
    'huaceproperty7d',
    'huaceproperty90d',
    'huaceproperty180d',
    'huaceshopping7d',
    'huaceshopping90d',
    'huaceshopping180d',
    'huacetravel7d',
    'huacetravel90d',
    'rn_hc',
    'smartscore',
    'rn_zm',
    'panshiscore',
    'rn_ps',
    'jdscore',
    'jdriskperiodpayment',
    'rn_jd',
    'xinyanmaxoverdueamt',
    'xinyanmaxoverduedays',
    'xinyancurrentlyoverduecount',
    'xinyancurrentlyperformance',
    'xinyancode',
    'latestoverduetime',
    'rn_xyb',
    'xinyanapplycredibility',
    'xinyanapplyscore',
    'xinyanapplylatestquerytime',
    'xinyanapplylatestsixmonth',
    'xinyanapplylatestthreemonth',
    'xinyanapplyquerycashcount',
    'xinyanapplyqueryfinancecount',
    'xinyanapplyqueryorgcount',
    'xinyanapplylatestonemonth',
    'xinyanapplyquerysumcount',
    'xinyanbehaviorconsfinorgcount',
    'xinyanbehaviorhistoryfailfee',
    'xinyanbehaviorhistorysucfee',
    'xinyanbehaviorlatestonemonth',
    'xinyanbehaviorlatestonemonthfail',
    'xinyanbehaviorlatestonemonthsuc',
    'xinyanbehaviorlatestsixmonth',
    'xinyanbehaviorlatestthreemonth',
    'xinyanbehaviorloanscount',
    'xinyanbehaviorloanscredibility',
    'xinyanbehaviorloanslatesttime',
    'xinyanbehaviorloanslongtime',
    'xinyanbehaviorloansorgcount',
    'xinyanbehaviorloansoverduecount',
    'xinyanbehaviorloansscore',
    'xinyanbehaviorloanssettlecount',
    'xinyancurrentconsfinavglimit',
    'xinyancurrentconsfincredibility',
    'xinyancurrentconsfincreditlimit',
    'xinyancurrentconsfinmaxlimit',
    'xinyancurrentconsfinorgcount',
    'xinyancurrentconsfinproductcount',
    'xinyancurrentloansavglimit',
    'xinyancurrentloanscredibility',
    'xinyancurrentloanscreditlimit',
    'xinyancurrentloansmaxlimit',
    'xinyancurrentloansorgcount',
    'xinyancurrentloansproductcount',
    'shumei_itfin_registers',
    'shumei_itfin_loan_applications',
    'shumei_itfin_loan_refuses',
    'shumei_itfin_loan_overdues',
    'shumei_itfin_loan_approvals',
    'shumei_itfin_loan_approval_level',
    'shumei_itfin_loan_overdue_duration',
    'shumei_itfin_loan_overdue_level',
    'shumei_debit_salary_level',
    'shumei_credit_registers',
    'shumei_credit_loan_applications',
    'shumei_credit_loan_approvals',
    'shumei_credit_loan_refuses',
    'shumei_credit_loan_overdues',
    'shumei_credit_loan_approval_level',
    'shumei_credit_loan_overdue_level',
    'shumei_credit_loan_overdue_duration',
    'rn_sm',
    'effectivenum',
    'totalnum',
    'lasttwoweekscallteltotalnums',
    'lasttwoweekscalltotaltimes',
    'lasttwoweekscallouttimes',
    'lasttwoweekscallintimes',
    'lasttwoweekscallavgduration',
    'lasttwoweekscalloutduration',
    'lasttwoweekscallinduration',
    'lasttwoweekscalldurationbelow15',
    'lasttwoweekscalldurationbetween15and30',
    'lasttwoweekscalldurationabove60',
    'last30dayscallteltotalnums',
    'last30dayscalltotaltimes',
    'last30dayscallouttimes',
    'last30dayscallintimes',
    'last30dayscallavgduration',
    'last30dayscalloutduration',
    'last30dayscallinduration',
    'last30dayscalldurationbelow15',
    'last30dayscalldurationbetween15and30',
    'last30dayscalldurationabove60',
    'overviewcallteltotalnums',
    'overviewcalltotaltimes',
    'overviewcallouttimes',
    'overviewcallintimes',
    'overviewcallavgduration',
    'overviewcalloutduration',
    'overviewcallinduration',
    'overviewcalldurationbelow15',
    'overviewcalldurationbetween15and30',
    'overviewcalldurationabove60',
    'overviewfirstcalltime',
    'overviewlastcalltime',
    'tongdunscore',
    'rn_td',
    'tdidnumber7dayscrossplatform',
    'tdidnumber30dayscrossplatform',
    'tdidnumber3monthscrossplatform',
    'tdmobile7dayscrossplatform',
    'tdmobile30dayscrossplatform',
    'tdmobile3monthscrossplatform',
    'tdidnumberp2p7dayscrossplatform',
    'tdidnumberp2p30dayscrossplatform',
    'tdidnumberp2p3monthscrossplatform',
    'tdmobilep2p7dayscrossplatform',
    'tdmobilep2p30dayscrossplatform',
    'tdmobilep2p3monthscrossplatform',
    'tdidnumberloan7dayscrossplatform',
    'tdidnumberloan30dayscrossplatform',
    'tdidnumberloan3monthscrossplatform',
    'tdmobileloan7dayscrossplatform',
    'tdmobileloan30dayscrossplatform',
    'tdmobileloan3monthscrossplatform',
    'tdidnumberhujin7dayscrossplatform',
    'tdidnumberhujin30dayscrossplatform',
    'tdidnumberhujin3monthscrossplatform',
    'tdmobilehujin7dayscrossplatform',
    'tdmobilehujin30dayscrossplatform',
    'tdmobilehujin3monthscrossplatform',
    'tdidnumberbystage7dayscrossplatform',
    'tdidnumberbystage30dayscrossplatform',
    'tdidnumberbystage3monthscrossplatform',
    'tdmobilebystage7dayscrossplatform',
    'tdmobilebystage30dayscrossplatform',
    'tdmobilebystage3monthscrossplatform',
    'tdidnumbercar7dayscrossplatform',
    'tdidnumbercar30dayscrossplatform',
    'tdidnumbercar3monthscrossplatform',
    'tdmobilecar7dayscrossplatform',
    'tdmobilecar30dayscrossplatform',
    'tdmobilecar3monthscrossplatform',
    'tdisblacklist',
    'tdgreylevel',
    'rn_td2',
    'caller_call_count_1',
    'called_call_count_1',
    'called_people_count_1',
    'caller_call_count_anomaly_count_1',
    'caller_call_count_anomaly_count_5',
    'caller_call_people_anomaly_count_1',
    'caller_call_people_anomaly_count_5',
    'called_call_time_anomaly_count_1',
    'called_call_count_anomaly_count_1',
    'called_call_count_anomaly_count_3',
    'called_call_count_anomaly_count_5',
    'called_call_people_anomaly_count_1',
    'called_call_people_anomaly_count_2',
    'called_call_people_anomaly_count_3',
    'max_silence_day_1',
    'max_silence_day_4',
    'no_call_day_count_3',
    'no_call_day_count_4',
    'call_address_count_3',
    'call_address_count_4',
    'call_address_count_6',
    'caller_call_count_2_4_4',
    'caller_call_count_24_2_2',
    'caller_call_count_24_2_4',
    'caller_call_count_24_2_5',
    'called_call_count_24_2_2',
    'called_call_count_24_2_4',
    'called_call_count_24_2_6',
    'called_call_count_2_4_4',
    'called_call_count_2_4_5',
    'caller_call_people_24_2_4',
    'caller_call_people_24_2_6',
    'called_max_call_count_daily_1',
    'called_max_call_count_daily_3',
    'called_max_call_people_daily_1',
    'called_max_call_people_daily_3',
    'score_behavior',
    'score_object',
    'score_all',
    'rn_cr',
]


'''
空值比例大于80%得  非名单类得一律踢除掉
 ['shumei_itfin_loan_overdue_duration' '0.9806706926525757' '845']
 ['shumei_credit_loan_overdue_duration' '0.9967975112087107' '140']
 ['huaceproperty7d' '0.9999542501601244' '2']
 ['huaceproperty90d' '0.9999542501601244' '2']
 ['huaceproperty180d' '0.9999542501601244' '2']
 ['huaceshopping7d' '0.9999542501601244' '2']
 ['huacetravel90d' '0.9999542501601244' '2']
 ['huaceshopping180d' '0.9999542501601244' '2']
 ['huacetravel7d' '0.9999542501601244' '2']
 ['rn_hc' '0.9999542501601244' '2']
 ['huaceloan180d' '0.9999542501601244' '2']
 ['huaceshopping90d' '0.9999542501601244' '2']
 ['huaceloan90d' '0.9999542501601244' '2']
 ['huacefinance7d' '0.9999542501601244' '2']
 ['huacelaunchday' '0.9999542501601244' '2']
 ['huaceappstability7d' '0.9999542501601244' '2']
 ['huaceappstability90d' '0.9999542501601244' '2']
 ['huaceappstability180d' '0.9999542501601244' '2']
 ['huaceloan7d' '0.9999542501601244' '2']
 ['huacecar90d' '0.9999542501601244' '2']
 ['huacecar7d' '0.9999542501601244' '2']
 ['huacedevicebrand' '0.9999542501601244' '2']
 ['huacedeviceprice' '0.9999542501601244' '2']
 ['huacedevicerank' '0.9999542501601244' '2']
 ['huacefinance90d' '0.9999542501601244' '2']
 ['huacefinance180d' '0.9999542501601244' '2']
 ['huacecar180d' '0.9999542501601244' '2']
 ['baiduidphonequeryorgcnty1' '1.0' '0']
 ['baiduidqueryorgcnty1' '1.0' '0']
 ['baiduidphonequerytimesy1' '1.0' '0']
 ['baiduidquerytimesy1' '1.0' '0']
 ['car_no' '1.0' '0']]
'''



filterAttribute=[
    'id',
    'shumei_itfin_loan_overdue_duration',
    'shumei_credit_loan_overdue_duration',
    'huaceproperty7d',
    'huaceproperty90d',
    'huaceproperty180d',
    'huaceshopping7d',
    'huacetravel90d',
    'huaceshopping180d',
    'huacetravel7d',
    'rn_hc',
    'huaceloan180d',
    'huaceshopping90d',
    'huaceloan90d',
    'huacefinance7d',
    'huacelaunchday',
    'huaceappstability7d',
    'huaceappstability90d',
    'huaceappstability180d',
    'huaceloan7d',
    'huacecar90d',
    'huacecar7d',
    'huacedevicebrand',
    'huacedeviceprice',
    'huacedevicerank',
    'huacefinance90d',
    'huacefinance180d',
    'baiduidphonequeryorgcnty1',
    'baiduidqueryorgcnty1',
    'baiduidphonequerytimesy1',
    'baiduidquerytimesy1',
    'car_no',
    'max_creditlimit_amount'
]

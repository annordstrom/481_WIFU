/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package pkg481.gui;

import java.util.Properties;

import javax.mail.Authenticator;
import javax.mail.Message;
import javax.mail.MessagingException;
import javax.mail.PasswordAuthentication;
import javax.mail.Session;
import javax.mail.Transport;
import javax.mail.internet.InternetAddress;
import javax.mail.internet.MimeMessage;

public class JavaMailUtil {
	public static void sendMail(String recepient, String user) throws MessagingException{
		System.out.println("Preparing to send email");
		Properties properties = new Properties();
		
		properties.put("mail.smtp.auth", "true");
		properties.put("mail.smtp.starttls.enable", "true");
		properties.put("mail.smtp.host", "smtp.gmail.com");
		properties.put("mail.smtp.port", "587");

		String myAccountEmail = "wirelessinhomefalldetection@gmail.com";
		String myAccountPassword = "WIFU2019";
		
		Session session = Session.getInstance(properties, new Authenticator() {
			@Override
			protected PasswordAuthentication getPasswordAuthentication() {
				return new PasswordAuthentication(myAccountEmail, myAccountPassword);
			}
		});
		
		Message message = prepareMessage(session, myAccountEmail, recepient, user);
		
		Transport.send(message);
		System.out.println("Message sent successfully");
		
	}
	
	private static Message prepareMessage(Session session, String myAccountEmail, String recepient, String user) {
			Message message = new MimeMessage(session);
			try {
				
				message.setFrom(new InternetAddress(myAccountEmail));
				message.setRecipient(Message.RecipientType.TO, new InternetAddress(recepient));
				message.setSubject("A Fall Has Been Detected");
				message.setText("You are the caregiver for " + user + " and it was detected that a fall has occurred");
				return message;
			} catch (MessagingException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			return null;
	
	}

}
